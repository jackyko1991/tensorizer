"""The solvers: a tensor PDLP for the LP relaxation, rounding + greedy repair to integers, and Hungarian matching.

PDLP (Applegate et al. 2021) is the primal–dual hybrid gradient method for LP with diagonal (Ruiz) preconditioning,
averaging, adaptive restarts and a primal weight; the same method runs inside Google OR-Tools and NVIDIA cuOpt.
Each iteration costs two sparse matrix–vector products, so it runs unchanged on CPU or CUDA.
"""

import math
import time
import warnings

import torch

from problems._common.progress import report

from . import equation
from .tensors import Cuts


def _ruiz(idx: torch.Tensor, val: torch.Tensor, m: int, n: int, iters: int = 12):
    """Row and column scalings D_r (M,), D_c (V,) that equilibrate |D_r A D_c| to max 1 in every row and column."""
    dr = torch.ones(m, dtype=val.dtype, device=val.device)
    dc = torch.ones(n, dtype=val.dtype, device=val.device)
    for _ in range(iters):
        v = (val * dr[idx[0]] * dc[idx[1]]).abs()
        rmax = torch.zeros_like(dr).scatter_reduce(0, idx[0], v, "amax")
        cmax = torch.zeros_like(dc).scatter_reduce(0, idx[1], v, "amax")
        dr = dr / torch.where(rmax > 0, rmax.sqrt(), torch.ones_like(rmax))
        dc = dc / torch.where(cmax > 0, cmax.sqrt(), torch.ones_like(cmax))
    return dr, dc


def _norm2(a: torch.Tensor, at: torch.Tensor, iters: int = 40) -> float:
    """Largest singular value of A by power iteration on AᵀA."""
    v = torch.ones(a.shape[1], dtype=torch.float64, device=a.device) / math.sqrt(a.shape[1])
    s = 1.0
    for _ in range(iters):
        w = at @ (a @ v)
        s = w.norm().item()
        v = w / max(s, 1e-300)
    return math.sqrt(s)


def kkt(a, at, b, c, z, y) -> dict:
    """Optimality measures of (z, y) for  min qᵀz, q = −c,  A z ≤ b,  0 ≤ z ≤ 1,  y ≥ 0  (unscaled).

    Reduced cost λ = q + Aᵀy. Dual objective D = −bᵀy + Σ min(λ, 0) is a lower bound for every y ≥ 0 because the
    box is finite. primal: ‖(Az − b)₊‖; dual: ‖λ⁺z + λ⁻(1 − z)‖ (reduced costs that disagree with the bounds);
    gap: |P − D|. All three are 0 exactly at an optimum (complementary slackness)."""
    q = -c
    lam = q + at @ y  # (V,)
    p_obj = (q @ z).item()
    d_obj = (-b @ y + lam.clamp(max=0).sum()).item()
    primal = (a @ z - b).clamp(min=0).norm().item() / (1 + b.norm().item())
    dual = (lam.clamp(min=0) * z + (-lam).clamp(min=0) * (1 - z)).norm().item() / (1 + c.norm().item())
    gap = abs(p_obj - d_obj) / (1 + abs(p_obj) + abs(d_obj))
    return {"primal": primal, "dual": dual, "gap": gap, "objective": -p_obj, "dual_bound": -d_obj}


def pdlp(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, max_iter: int = 5000, tol: float = 1e-5, check: int = 64):
    """Solve  max cᵀz  s.t.  A z ≤ b,  0 ≤ z ≤ 1  by restarted, preconditioned PDHG (PDLP).

    Scaled problem Â = D_r A D_c, ẑ = z/D_c with box [0, 1/D_c]. One iteration:
        ẑ⁺ = clip(ẑ − τ(q̂ + Âᵀŷ), 0, û),     ŷ⁺ = max(0, ŷ + σ(Â(2ẑ⁺ − ẑ) − b̂)),    τ = η/ω, σ = ηω, η‖Â‖ < 1.
    Every `check` iterations the current and averaged iterates are scored by their KKT error; restart to the better
    one on sufficient (×0.2) or necessary (×0.8, no longer improving) decay, then update the primal weight ω.
    Returns z, y, objective, dual_bound, history (iteration, primal, dual, gap) and snapshots of z."""
    t0 = time.perf_counter()
    warnings.filterwarnings("ignore", message="Sparse (CSR tensor support|invariant checks)")
    a = a.coalesce()
    m, n = a.shape
    idx, val = a.indices(), a.values().double()
    dr, dc = _ruiz(idx, val, m, n)
    a_hat = torch.sparse_coo_tensor(idx, val * dr[idx[0]] * dc[idx[1]], (m, n)).coalesce().to_sparse_csr()
    at_hat = a_hat.t().to_sparse_csr()
    a_csr, at_csr = a.to_sparse_csr(), a.t().coalesce().to_sparse_csr()
    q_hat, b_hat, u_hat = -c * dc, b * dr, 1.0 / dc
    eta = 0.95 / _norm2(a_hat, at_hat)
    bn, qn = b_hat.norm().item(), q_hat.norm().item()
    omega = qn / bn if bn > 1e-12 and qn > 1e-12 else 1.0
    z = torch.zeros(n, dtype=torch.float64, device=c.device)
    y = torch.zeros(m, dtype=torch.float64, device=c.device)
    z_sum, y_sum, n_avg = torch.zeros_like(z), torch.zeros_like(y), 0
    z_last, y_last, kkt_last, kkt_prev = z.clone(), y.clone(), math.inf, math.inf
    history, snaps = [], []

    def score(zh, yh):
        m_ = kkt(a_csr, at_csr, b, c, dc * zh, dr * yh)
        return m_, math.sqrt(omega**2 * m_["primal"] ** 2 + m_["dual"] ** 2 / omega**2 + m_["gap"] ** 2), m_

    best = None
    for k in range(1, max_iter + 1):
        tau, sigma = eta / omega, eta * omega
        z_new = torch.minimum(torch.clamp(z - tau * (q_hat + at_hat @ y), min=0), u_hat)
        y = torch.clamp(y + sigma * (a_hat @ (2 * z_new - z) - b_hat), min=0)
        z = z_new
        z_sum += z
        y_sum += y
        n_avg += 1
        if k % check and k != max_iter:
            continue
        m_cur, k_cur, _ = score(z, y)
        m_avg, k_avg, _ = score(z_sum / n_avg, y_sum / n_avg)
        use_avg = k_avg < k_cur
        cand_z, cand_y = (z_sum / n_avg, y_sum / n_avg) if use_avg else (z, y)
        k_cand, m_cand = (k_avg, m_avg) if use_avg else (k_cur, m_cur)
        best = (dc * cand_z, dr * cand_y, m_cand)
        history.append((k, m_cand["primal"], m_cand["dual"], m_cand["gap"], m_cand["objective"]))
        snaps.append((k, (dc * cand_z).clamp(0, 1).clone()))
        report(k, max_iter, "PDLP iterations")
        if max(m_cand["primal"], m_cand["dual"], m_cand["gap"]) < tol:
            break
        if k_cand <= 0.2 * kkt_last or (k_cand <= 0.8 * kkt_last and k_cand > kkt_prev) or n_avg >= 0.36 * k:
            dz, dy = (cand_z - z_last).norm().item(), (cand_y - y_last).norm().item()
            if dz > 1e-10 and dy > 1e-10:  # primal weight: balance the distance travelled in z and in y
                omega = math.exp(0.5 * math.log(dy / dz) + 0.5 * math.log(omega))
            z, y = cand_z.clone(), cand_y.clone()
            z_last, y_last, kkt_last = z.clone(), y.clone(), k_cand
            z_sum, y_sum, n_avg = torch.zeros_like(z), torch.zeros_like(y), 0
        kkt_prev = k_cand
    z_u, y_u, m_fin = best
    hist = torch.tensor(history, dtype=torch.float64)
    return {
        "z": z_u.clamp(0, 1),
        "y": y_u,
        "objective": m_fin["objective"],
        "dual_bound": m_fin["dual_bound"],
        "iterations": k,
        "history": {"iter": hist[:, 0], "primal": hist[:, 1], "dual": hist[:, 2], "gap": hist[:, 3], "obj": hist[:, 4]},
        "snaps": snaps,
        "seconds": time.perf_counter() - t0,
    }


def round_and_repair(z, seg, trip, score, n_hits: int, lam: float = Cuts.lam):
    """Integer (x (T,), y (E,)) from an LP point z = (x, y): rounding then a greedy conflict repair, in parallel.

    Segments are ranked by LP value y_e (ties: best triplet score). Each round accepts every segment that ranks
    first among the undecided segments leaving its inner hit AND entering its outer hit, then discards segments
    that clash with an accepted one; this is exactly sequential greedy, done as tensor rounds. Triplets are kept
    when both segments are accepted; segments that carry no kept triplet are dropped (they only cost λ), and chain
    ends whose last triplet scores below λ (a lone triplet below 2λ) are trimmed."""
    n_t, n_e, dev = trip.shape[0], seg.shape[0], seg.device
    y_lp = z[n_t:]
    best_s = torch.zeros(n_e, dtype=torch.float64, device=dev)
    best_s.scatter_reduce_(0, trip.reshape(-1), score.double().repeat_interleave(2), "amax")
    useful = torch.zeros(n_e, dtype=torch.bool, device=dev)
    useful[trip.reshape(-1)] = True
    order = torch.argsort(y_lp + 1e-3 * best_s, stable=True)
    rank = torch.empty(n_e, dtype=torch.float64, device=dev)
    rank[order] = torch.arange(n_e, dtype=torch.float64, device=dev)  # unique priorities
    undecided, chosen = useful.clone(), torch.zeros(n_e, dtype=torch.bool, device=dev)
    busy_out = torch.zeros(n_hits, dtype=torch.bool, device=dev)
    busy_in = torch.zeros(n_hits, dtype=torch.bool, device=dev)
    while undecided.any():
        p = torch.where(undecided, rank, torch.full_like(rank, -1.0))
        top_out = torch.full((n_hits,), -1.0, dtype=torch.float64, device=dev).scatter_reduce(0, seg[:, 0], p, "amax")
        top_in = torch.full((n_hits,), -1.0, dtype=torch.float64, device=dev).scatter_reduce(0, seg[:, 1], p, "amax")
        win = undecided & (p == top_out[seg[:, 0]]) & (p == top_in[seg[:, 1]])
        chosen |= win
        busy_out[seg[win, 0]] = True
        busy_in[seg[win, 1]] = True
        undecided &= ~win & ~busy_out[seg[:, 0]] & ~busy_in[seg[:, 1]]
    while True:  # trim chain ends whose last triplet is worth less than the segments it needs
        x = chosen[trip[:, 0]] & chosen[trip[:, 1]]
        y = torch.zeros(n_e, dtype=torch.bool, device=dev)
        y[trip[x].reshape(-1)] = True
        as_first = torch.zeros(n_e, dtype=torch.bool, device=dev)
        as_second = torch.zeros(n_e, dtype=torch.bool, device=dev)
        as_first[trip[x, 0]] = True
        as_second[trip[x, 1]] = True
        weak = x & (score < lam)
        drop_out = weak & ~as_first[trip[:, 1]]  # outer segment ends the chain
        drop_in = weak & ~as_second[trip[:, 0]]  # inner segment starts the chain
        alone = x & ~as_first[trip[:, 1]] & ~as_second[trip[:, 0]] & (score < 2 * lam)  # a lone triplet pays 2λ
        drop = torch.zeros(n_e, dtype=torch.bool, device=dev)
        drop[trip[drop_out, 1]] = True
        drop[trip[drop_in & ~drop_out, 0]] = True
        drop[trip[alone].reshape(-1)] = True
        if not drop.any():
            return x.long(), y.long()
        chosen = y & ~drop


def hungarian(cost: torch.Tensor) -> torch.Tensor:
    """Exact minimum-cost assignment (Kuhn–Munkres with potentials, O(n² m)): column of each row, (n,), n ≤ m.

    Rows are added one at a time; a shortest augmenting path (Dijkstra on reduced costs c_ij − u_i − v_j ≥ 0) is
    grown over all columns at once with vector operations, then the potentials u, v are updated by the step δ."""
    n, m = cost.shape
    if n > m:
        col_of_row_t = hungarian(cost.t())  # (m,) row of each column
        out = torch.empty(n, dtype=torch.long, device=cost.device).fill_(-1)
        out[col_of_row_t] = torch.arange(m, device=cost.device)
        return out
    c = cost.double()
    inf = torch.tensor(float("inf"), dtype=torch.float64, device=c.device)
    u = torch.zeros(n + 1, dtype=torch.float64, device=c.device)
    v = torch.zeros(m + 1, dtype=torch.float64, device=c.device)
    p = torch.zeros(m + 1, dtype=torch.long, device=c.device)  # p[j]: row matched to column j (1-based, 0 = free)
    way = torch.zeros(m + 1, dtype=torch.long, device=c.device)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = torch.full((m + 1,), float("inf"), dtype=torch.float64, device=c.device)
        used = torch.zeros(m + 1, dtype=torch.bool, device=c.device)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            cur = c[i0 - 1] - u[i0] - v[1:]  # (m,) reduced costs from row i0
            better = ~used[1:] & (cur < minv[1:])
            minv[1:] = torch.where(better, cur, minv[1:])
            way[1:] = torch.where(better, torch.tensor(j0, device=c.device), way[1:])
            masked = torch.where(used[1:], inf, minv[1:])
            j1 = int(torch.argmin(masked)) + 1
            delta = masked[j1 - 1]
            u.index_add_(0, p[used], delta.expand(int(used.sum())))
            v = torch.where(used, v - delta, v)
            minv = torch.where(used, minv, minv - delta)
            j0 = j1
            if int(p[j0]) == 0:
                break
        while j0:  # augment along the path
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    col = torch.empty(n, dtype=torch.long, device=c.device)
    matched = p[1:] > 0
    col[p[1:][matched] - 1] = torch.arange(m, device=c.device)[matched]
    return col


def hungarian_tracks(xyz, layer, seg, cuts: Cuts, assign=None) -> torch.Tensor:
    """Baseline: layer-to-layer linear assignment problems, chained into tracks. Returns a bool mask (E,).

    Each step matches hits on layer l to at most one hit on layer l+1 (a 'no match' column costs 1). The middle
    layer pair goes first with cost 1 − (best triplet score the segment could join); then the inner pair, whose cost
    is 1 − s_t of the triplet with the already chosen continuation, then the outer pair the same way (curvature and
    angle mismatch over three layers, but decided one layer pair at a time). Chains shorter than 3 hits are dropped.
    `assign` swaps in another exact assignment solver (e.g. scipy's linear_sum_assignment, offline)."""
    assign = assign or hungarian
    n, n_e, dev = xyz.shape[0], seg.shape[0], xyz.device
    trip_all = equation.join(seg, n)
    s_all = equation.triplet_score(xyz, seg, trip_all, cuts)
    trip_all, s_all = trip_all[s_all > 0], s_all[s_all > 0]
    support = torch.zeros(n_e, dtype=torch.float64, device=dev)
    support.scatter_reduce_(0, trip_all.reshape(-1), s_all.repeat_interleave(2), "amax")
    pred = torch.full((n,), -1, dtype=torch.long, device=dev)  # chosen segment entering each hit
    succ = torch.full((n,), -1, dtype=torch.long, device=dev)  # chosen segment leaving each hit
    chosen = torch.zeros(n_e, dtype=torch.bool, device=dev)
    n_pairs = int(layer.max()) if n else 0
    first = n_pairs // 2
    for lay in [first, *range(first - 1, -1, -1), *range(first + 1, n_pairs)]:  # middle, inwards, outwards
        es = (layer[seg[:, 0]] == lay).nonzero().squeeze(1)
        if es.numel() == 0:
            continue
        cost_e = 1 - support[es]
        if lay != first:
            cost_e = torch.ones_like(cost_e)
            mine = torch.zeros(n_e, dtype=torch.bool, device=dev)
            mine[es] = True
            pos = torch.full((n_e,), -1, dtype=torch.long, device=dev)
            pos[es] = torch.arange(es.numel(), device=dev)
            if lay < first:  # triplet (e, chosen continuation of e's outer hit)
                ok = mine[trip_all[:, 0]] & (trip_all[:, 1] == succ[seg[trip_all[:, 0], 1]])
                cost_e[pos[trip_all[ok, 0]]] = 1 - s_all[ok]
            else:  # triplet (chosen segment into e's inner hit, e)
                ok = mine[trip_all[:, 1]] & (trip_all[:, 0] == pred[seg[trip_all[:, 1], 0]])
                cost_e[pos[trip_all[ok, 1]]] = 1 - s_all[ok]
        rows, ri = torch.unique(seg[es, 0], return_inverse=True)
        cols, ci = torch.unique(seg[es, 1], return_inverse=True)
        cost = torch.full((rows.numel(), cols.numel() + rows.numel()), 10.0, dtype=torch.float64, device=dev)
        cost[:, cols.numel() :] = 1.0  # 'no match' columns
        cost[ri, ci] = cost_e.clamp(max=10.0)
        col = assign(cost)
        real = (col < cols.numel()) & (cost[torch.arange(rows.numel(), device=dev), col] < 1.0)
        key = torch.full((rows.numel(), cols.numel()), -1, dtype=torch.long, device=dev)
        key[ri, ci] = es
        picked = key[real.nonzero().squeeze(1), col[real]]
        chosen[picked] = True
        pred[seg[picked, 1]] = picked
        succ[seg[picked, 0]] = picked
    label = equation.tracks_from_segments(seg[chosen], n)
    size = torch.bincount(label[label >= 0], minlength=n)
    return chosen & (size[label[seg[:, 0]].clamp(min=0)] >= 3)


def triplets_of(chosen_seg: torch.Tensor, trip: torch.Tensor) -> torch.Tensor:
    """x_t = 1 when both segments of triplet t are chosen (T,)."""
    return (chosen_seg[trip[:, 0]] & chosen_seg[trip[:, 1]]).long()


def fit_scaling(size: torch.Tensor, seconds: torch.Tensor) -> tuple[float, float]:
    """Least squares in log–log: t ≈ C nᵏ. Returns (k, C)."""
    x, y = size.double().log(), seconds.double().log()
    design = torch.stack([x, torch.ones_like(x)], 1)
    k, logc = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
    return k.item(), math.exp(logc.item())
