"""The pipeline stages, each a batched tensor function over the case axis (B cases at once, or B = 1 when Snakemake
runs one job), and the tensor workflow engine that reads the Snakefile and runs the whole DAG.

Stages:  segment (soft threshold + connected components) → centreline (min-plus eikonal on the labelled voxel graph)
         → measure (radii from the soft segmentation, lengths, asymmetry, stenosis) → boundaries (ICA/BA inlets,
         MCA/ACA/PCA outlets) → flow (Poiseuille network + Murray loads, one batched linear solve) → report → cohort.
Only torch is imported here, so the same file runs inside each Snakemake job.
"""

import math
import re
import time

import torch
import torch.nn.functional as F

from . import equation as eq
from . import tensors as T

MIN_CC = 64  # connected components smaller than this (voxels, 8 mm³) are noise
CLUSTER = 3.5e-3  # label contacts closer than this (m) are one junction
OFFSETS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]  # 13


# ---------------------------------------------------------------- voxel-graph helpers
def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    """Cube dilation by r voxels, (B, X, Y, Z) bool."""
    return F.max_pool3d(mask.double()[:, None], 2 * r + 1, 1, r)[:, 0] > 0


def distance_map(mask: torch.Tensor, max_r: int = 16) -> torch.Tensor:
    """Depth of every voxel in the mask (m): erosions survived, alternating cube and cross elements (an octagonal
    approximation of the Euclidean distance to the wall); a voxel on the wall gets h/2."""
    x = mask.double()[:, None]
    d = torch.zeros_like(x)
    for k in range(max_r):
        d = d + x
        if k % 2 == 0:
            x = -F.max_pool3d(-x, 3, 1, 1)  # cube erosion (outside the crop counts as inside: no erosion there)
        else:
            xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
            X, Y, Z = x.shape[2:]
            y = x
            for a, b, c in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):  # cross erosion
                y = torch.minimum(y, xp[:, :, a : a + X, b : b + Y, c : c + Z])
            x = y
        if not bool(x.any()):
            break
    return ((d - 0.5).clamp(min=0) * T.H)[:, 0]


def voxel_graph(mask: torch.Tensor):
    """Voxels of a (B, X, Y, Z) mask as nodes (coords (V, 4): case, i, j, k, ordered case by case) and their
    26-neighbour pairs (src, dst) once each, with the edge lengths (m)."""
    coords = mask.nonzero()
    index = torch.full(mask.shape, -1, dtype=torch.long)
    index[mask] = torch.arange(len(coords))
    shape = torch.tensor(mask.shape[1:])
    src, dst, ln = [], [], []
    for o in OFFSETS:
        n = coords[:, 1:] + torch.tensor(o)
        ok = ((n >= 0) & (n < shape)).all(1)
        nb = torch.full((len(coords),), -1, dtype=torch.long)
        nb[ok] = index[coords[ok, 0], n[ok, 0], n[ok, 1], n[ok, 2]]
        keep = nb >= 0
        src.append(torch.arange(len(coords))[keep]), dst.append(nb[keep])
        ln.append(torch.full((int(keep.sum()),), math.sqrt(sum(c * c for c in o)) * T.H, dtype=torch.float64))
    return coords, torch.cat(src), torch.cat(dst), torch.cat(ln)


def components(n: int, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Connected components: every node takes the smallest node index of its component (min-label propagation with
    pointer jumping), (n,) long."""
    lab = torch.arange(n)
    while True:
        m = torch.minimum(lab[src], lab[dst])
        new = lab.scatter_reduce(0, src, m, "amin").scatter_reduce(0, dst, m, "amin")
        new = new[new]
        if torch.equal(new, lab):
            return lab
        lab = new


def eikonal(n: int, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor, sources: torch.Tensor):
    """Arrival times from the source nodes over undirected edges by min-plus sweeps, and each node's predecessor
    on a shortest path (itself for sources and unreached nodes)."""
    s2, d2, w2 = torch.cat([src, dst]), torch.cat([dst, src]), torch.cat([w, w])
    Tt = torch.full((n,), math.inf, dtype=torch.float64)
    Tt[sources] = 0.0
    while True:
        new = eq.relax(Tt, s2, d2, w2)
        if torch.equal(new, Tt):
            break
        Tt = new
    tight = (Tt[s2] + w2 == Tt[d2]) & torch.isfinite(Tt[d2])
    pred = torch.arange(n).scatter_reduce(0, d2[tight], s2[tight], "amin", include_self=False)
    pred[sources] = sources
    return Tt, pred


def per_case_rank(b: torch.Tensor, B: int) -> torch.Tensor:
    """Position of each item within its case, for items sorted by case: (n,) long."""
    counts = torch.bincount(b, minlength=B)
    start = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    return torch.arange(len(b)) - start[b]


def segment_argmin(values: torch.Tensor, group: torch.Tensor, n_groups: int) -> torch.Tensor:
    """For each group, the index of its smallest value (ties: the first index); −1 for an empty group."""
    best = torch.full((n_groups,), math.inf, dtype=values.dtype).scatter_reduce(0, group, values, "amin")
    hit = values == best[group]
    idx = torch.full((n_groups,), len(values), dtype=torch.long)
    idx = idx.scatter_reduce(0, group[hit], torch.arange(len(values))[hit], "amin")
    return torch.where(idx == len(values), -1, idx)


# ---------------------------------------------------------------- stage 1: segment
def segment(image: torch.Tensor, labels: torch.Tensor, theta: torch.Tensor, width: float = T.WIDTH, keep=None):
    """Threshold segmentation of the MRA: soft s = σ((I − θ)/w) (differentiable in θ), the hard mask I > θ minus
    components under MIN_CC voxels, its Dice against the TopCoW mask, and the region within 2 voxels of the labelled
    vessels (where the vessel calibre is measured). `keep` (the component mask) can be held fixed."""
    soft = eq.soft_threshold(image, theta, width)
    hard = image > theta.detach()[:, None, None, None]
    if keep is None:
        coords, src, dst, _ = voxel_graph(hard)
        lab = components(len(coords), src, dst)
        size = torch.bincount(lab, minlength=len(coords))[lab]
        keep = torch.zeros_like(hard)
        keep[hard] = size >= MIN_CC
        keep = dilate(keep, 1)  # room for the soft rim around each kept component
    truth = labels > 0
    mask = hard & keep
    return {"soft": soft * keep, "mask": mask, "keep": keep, "region": dilate(truth, 2), "dice": eq.dice(mask, truth)}


# ---------------------------------------------------------------- stage 2: centreline graph
def centreline(labels: torch.Tensor) -> dict:
    """Centreline graph of each case from the TopCoW labels. Junction nodes where labelled vessels touch (contacts
    closer than CLUSTER merged), a free-end node per terminal vessel (its voxel farthest from its junctions, by a
    min-plus eikonal with travel time ℓ/R²), and one edge per vessel piece between consecutive nodes along the
    shortest path. All vessels of all cases form one disjoint voxel graph, so each eikonal is one batched loop."""
    B = labels.shape[0]
    vessel = labels > 0
    R = distance_map(vessel)
    coords, src, dst, ln = voxel_graph(vessel)
    V = len(coords)
    bv, lv = coords[:, 0], labels[vessel]
    pos = coords[:, 1:].double() * T.H  # (V, 3) voxel centres (m)
    Rv = R[vessel]

    # largest connected piece of each labelled vessel (an instance = one (case, label))
    same = lv[src] == lv[dst]
    comp = components(V, src[same], dst[same])
    size = torch.bincount(comp, minlength=V)
    key = bv * 16 + lv
    score = size[comp] * (V + 1) + comp
    best = torch.zeros(16 * B, dtype=torch.long).scatter_reduce(0, key, score, "amax")
    kept = score == best[key]
    ekeep = kept[src] & kept[dst]

    # contacts between different labels → junction clusters
    cross = ekeep & ~same
    cs, cd = src[cross], dst[cross]
    la, lb = torch.minimum(lv[cs], lv[cd]), torch.maximum(lv[cs], lv[cd])
    # only anatomical connections count: e.g. two A2s running side by side touch but do not communicate; the two
    # ACAs count as connected only in a case without an Acom label (a fused or very short Acom)
    pair = la * 16 + lb
    allowed = torch.isin(pair, torch.tensor([a * 16 + b for a, b in T.CONNECTIONS]))
    no_acom = ~torch.zeros(B, dtype=torch.bool).index_fill(0, bv[lv == 10].unique(), True)
    allowed |= (pair == 11 * 16 + 12) & no_acom[bv[cs]]
    cs, cd, la, lb = cs[allowed], cd[allowed], la[allowed], lb[allowed]
    ckey = bv[cs] * 256 + la * 16 + lb
    ukey, cinv = torch.unique(ckey, return_inverse=True)
    C = len(ukey)
    cnt = torch.bincount(cinv, minlength=C).double()
    cpos = torch.zeros(C, 3, dtype=torch.float64).index_add(0, cinv, (pos[cs] + pos[cd]) / 2) / cnt[:, None]
    cb, cla, clb = ukey // 256, (ukey // 16) % 16, ukey % 16
    near = (torch.cdist(cpos, cpos) < CLUSTER) & (cb[:, None] == cb[None, :])
    reach = eq.closure(near.double()) | torch.eye(C, dtype=torch.bool)  # transitive closure: one cluster each
    rep = reach.double().argmax(1)  # smallest contact index in the cluster
    jroots, cnode = torch.unique(rep, return_inverse=True)  # junction nodes (sorted by case, then first contact)
    NJ = len(jroots)
    jb = cb[jroots]
    jpos = torch.zeros(NJ, 3, dtype=torch.float64).index_add(0, cnode, cpos)
    jpos = jpos / torch.bincount(cnode, minlength=NJ).double()[:, None]

    # instances and their attached junctions
    inst_keys, vinst = torch.unique(key[kept], return_inverse=True)
    K = len(inst_keys)
    inst_of_key = torch.full((16 * B,), -1, dtype=torch.long)
    inst_of_key[inst_keys] = torch.arange(K)
    ib, il = inst_keys // 16, inst_keys % 16
    att = torch.unique(torch.cat([torch.stack([inst_of_key[cb * 16 + cla], cnode], 1),
                                  torch.stack([inst_of_key[cb * 16 + clb], cnode], 1)]), dim=0)  # fmt: skip
    att = att[att[:, 0] >= 0]
    a_inst, a_node = att[:, 0], att[:, 1]
    n_att = torch.bincount(a_inst, minlength=K)
    terminal = torch.isin(il, torch.tensor(T.TERMINAL))
    usable = torch.where(terminal, n_att >= 1, n_att >= 2)

    # the kept voxel graph (re-indexed), with eikonal weights ℓ/R²
    kidx = torch.full((V,), -1, dtype=torch.long)
    kidx[kept] = torch.arange(int(kept.sum()))
    ks, kd = kidx[src[ekeep & same]], kidx[dst[ekeep & same]]
    kpos, kR = pos[kept], Rv[kept]
    w = eq.edge_cost(ln[ekeep & same], (kR[ks] + kR[kd]) / 2)
    nk = len(kpos)

    # seed voxel of each attachment: the instance voxel nearest to the junction
    vox_by_inst = torch.argsort(vinst, stable=True)
    first = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(vinst, minlength=K).cumsum(0)])
    n_in = first[a_inst + 1] - first[a_inst]
    rep_a = torch.repeat_interleave(torch.arange(len(att)), n_in)
    within = torch.arange(int(n_in.sum())) - torch.repeat_interleave(torch.cumsum(n_in, 0) - n_in, n_in)
    cand = vox_by_inst[first[a_inst][rep_a] + within]
    dist = (kpos[cand] - jpos[a_node][rep_a]).norm(dim=1)
    a_seed = cand[segment_argmin(dist, rep_a, len(att))]

    # eikonal 1 from all junction seeds: each terminal vessel's free end is its farthest voxel along the vessel
    T1, _ = eikonal(nk, ks, kd, ln[ekeep & same], a_seed)  # plain path length: thin side branches lose
    T1i = torch.where(torch.isfinite(T1), T1, torch.tensor(-1.0, dtype=torch.float64))
    far = torch.full((K,), -1.0, dtype=torch.float64).scatter_reduce(0, vinst, T1i, "amax")
    hit = T1i == far[vinst]
    far_vox = torch.full((K,), nk, dtype=torch.long).scatter_reduce(0, vinst[hit], torch.arange(nk)[hit], "amin")
    # re-centre the end: the most central voxel (largest R) within 2.5 mm of the farthest one (a cap or a crop face)
    dfar = (kpos - kpos[far_vox.clamp(max=nk - 1)][vinst]).norm(dim=1)
    score = torch.where(dfar < 2.5e-3, kR - 1e-3 * dfar, torch.tensor(-math.inf, dtype=torch.float64))
    end_vox = segment_argmin(-score, vinst, K)
    # the first attachment of each instance (its connector start)
    first_att = torch.full((K,), len(att), dtype=torch.long).scatter_reduce(0, a_inst, torch.arange(len(att)), "amin")
    start_vox = torch.where(terminal, end_vox, a_seed[first_att.clamp(max=len(att) - 1)])
    use_k = usable.nonzero().flatten()

    # eikonal 2 from each instance's start: orders its junctions and gives the path back to the start
    T2, pred = eikonal(nk, ks, kd, w, start_vox[use_k])
    a_use = usable[a_inst] & ~(~terminal[a_inst] & (torch.arange(len(att)) == first_att[a_inst]))
    au = a_use.nonzero().flatten()  # attachments that become chain points after the start
    order = torch.argsort(a_inst[au] * 1e6 + T2[a_seed[au]] / T2[torch.isfinite(T2)].max().clamp(min=1e-30))
    au = au[order]  # by instance, then by distance from the start
    lastpos = torch.full((K,), -1, dtype=torch.long).scatter_reduce(0, a_inst[au], torch.arange(len(au)), "amax")
    last = torch.where(lastpos >= 0, au[lastpos.clamp(min=0)], -1)  # each instance's farthest junction
    # backtrack all instances at once from their farthest junction seed
    cur = a_seed[last[use_k]]
    paths = [cur]
    while True:
        nxt = pred[cur]
        if torch.equal(nxt, cur):
            break
        cur = nxt
        paths.append(cur)
    P = torch.stack(paths[::-1], 1)  # (Ku, Lmax) start … farthest seed, left-padded with the start voxel repeated
    pad_n = (P == P[:, :1]).sum(1) - 1  # leading repeats to drop
    Lmax = P.shape[1]
    shift = (torch.arange(Lmax)[None, :] + pad_n[:, None]).clamp(max=Lmax - 1)
    P = P.gather(1, shift)
    plen = Lmax - pad_n  # valid points per path
    pmask = torch.arange(Lmax)[None, :] < plen[:, None]

    # chain points: start, then each attachment's nearest path index
    ku_of = torch.full((K,), -1, dtype=torch.long)
    ku_of[use_k] = torch.arange(len(use_k))
    row = ku_of[a_inst[au]]
    d_pa = (kpos[P[row]] - kpos[a_seed[au]][:, None, :]).norm(dim=2)
    d_pa = torch.where(pmask[row], d_pa, torch.tensor(math.inf, dtype=torch.float64))
    cut = d_pa.argmin(1)
    cut = torch.where(au == last[a_inst[au]], plen[row] - 1, cut)  # the farthest one ends the path
    big = Lmax + 1  # cuts never go back along a path: running maximum within each instance
    cut = torch.cummax(a_inst[au] * big + cut, 0).values - a_inst[au] * big
    # node ids per case: junctions first, then one free-end node per usable terminal instance
    j_rank = per_case_rank(jb, B)
    nj_case = torch.bincount(jb, minlength=B)
    te = use_k[terminal[use_k]]
    e_rank = per_case_rank(ib[te], B)
    end_node_local = torch.full((K,), -1, dtype=torch.long)
    end_node_local[te] = nj_case[ib[te]] + e_rank
    start_node = torch.where(terminal, end_node_local, j_rank[a_node[first_att.clamp(max=len(att) - 1)]])
    # edges: (start → first attachment), (attachment → next attachment), … per instance
    prev_same = torch.cat([torch.tensor([False]), a_inst[au][1:] == a_inst[au][:-1]])
    u_node = torch.where(prev_same, torch.roll(j_rank[a_node[au]], 1), start_node[a_inst[au]])
    u_cut = torch.where(prev_same, torch.roll(cut, 1), torch.zeros_like(cut))
    v_node, v_cut = j_rank[a_node[au]], cut
    e_case, e_label = ib[a_inst[au]], il[a_inst[au]]

    # polylines between the cuts: path voxel centres, ends pinned to the node positions, lightly smoothed
    nseg = (v_cut - u_cut + 1).clamp(min=2)
    Pm = int(nseg.max())
    idx = (u_cut[:, None] + torch.arange(Pm)[None, :]).clamp(max=Lmax - 1)
    ok = torch.arange(Pm)[None, :] < nseg[:, None]
    idx = torch.minimum(idx, v_cut[:, None])
    poly = kpos[P[row].gather(1, idx)]  # (E, Pm, 3)
    n_total = nj_case + torch.bincount(ib[te], minlength=B)
    N = int(n_total.max())
    node_pos = torch.full((B, N, 3), math.nan, dtype=torch.float64)
    node_pos[jb, j_rank] = jpos
    node_pos[ib[te], end_node_local[te]] = kpos[end_vox[te]]
    node_kind = torch.full((B, N), -1, dtype=torch.long)
    node_kind[jb, j_rank] = 0
    node_kind[ib[te], end_node_local[te]] = il[te]
    poly[:, 0] = node_pos[e_case, u_node]
    last_i = (nseg - 1)[:, None, None].expand(-1, 1, 3)
    poly = poly.scatter(1, last_i, node_pos[e_case, v_node][:, None, :])
    poly = smooth_polyline(poly, ok)
    seg_len = torch.where(ok[:, 1:], (poly[:, 1:] - poly[:, :-1]).norm(dim=2), torch.zeros(1, dtype=torch.float64))
    radius_dt = torch.where(ok, kR[P[row].gather(1, idx)], torch.zeros(1, dtype=torch.float64)).sum(1) / nseg

    # pack per case: (B, E, …), padded
    er = per_case_rank(e_case, B)
    E = int(torch.bincount(e_case, minlength=B).max())
    edges = torch.zeros(B, E, 2, dtype=torch.long)
    edges[e_case, er] = torch.stack([u_node, v_node], 1)
    out_poly = torch.full((B, E, Pm, 3), math.nan, dtype=torch.float64)
    out_poly[e_case, er] = torch.where(ok[..., None], poly, torch.tensor(math.nan, dtype=torch.float64))
    pack = lambda x, fill: torch.full((B, E), fill, dtype=x.dtype).index_put((e_case, er), x)  # noqa: E731
    return {
        "node_pos": node_pos, "node_kind": node_kind, "node_valid": node_kind >= 0,
        "edges": edges, "edge_label": pack(e_label, 0),
        "edge_valid": pack(torch.ones_like(e_label, dtype=torch.bool), False),
        "poly": out_poly, "length": pack(seg_len.sum(1), 0.0), "radius_dt": pack(radius_dt, 0.0),
    }  # fmt: skip


def smooth_polyline(poly: torch.Tensor, ok: torch.Tensor, half: int = 2) -> torch.Tensor:
    """Moving average over ±half points along each polyline (E, P, 3), end points kept."""
    x = torch.where(ok[..., None], poly, torch.zeros(1, dtype=poly.dtype)).transpose(1, 2)  # (E, 3, P)
    k = torch.ones(1, 1, 2 * half + 1, dtype=poly.dtype)
    num = F.conv1d(x.reshape(-1, 1, x.shape[-1]), k, padding=half).reshape(x.shape)
    den = F.conv1d(ok.double()[:, None], k, padding=half)
    sm = (num / den.clamp(min=1)).transpose(1, 2)
    n = ok.sum(1)
    pin = (torch.arange(poly.shape[1])[None, :] == 0) | (torch.arange(poly.shape[1])[None, :] == (n - 1)[:, None])
    return torch.where(pin[..., None] | ~ok[..., None], poly, sm)


# ---------------------------------------------------------------- stage 3: measure
def measure(seg: dict, g: dict, chunk: int = 2048, near=None) -> dict:
    """Calibre of every edge from the soft segmentation: each voxel of the vessel region goes to its nearest
    centreline point, the soft volume per edge V_e = Σ s h³ gives r_e = √(V_e/(πL_e)), differentiable in θ; the same
    per point gives a radius profile (stenosis: a dip below half the median radius). Also L/R asymmetry per vessel."""
    soft, region = seg["soft"], seg["region"]
    B, E, Pm, _ = g["poly"].shape
    flat = g["poly"].reshape(B, E * Pm, 3)
    valid = ~torch.isnan(flat[..., 0])
    pb, pq = valid.nonzero(as_tuple=True)  # the real centreline points, packed per case (order kept)
    pr = per_case_rank(pb, B)
    pts = torch.full((B, int(torch.bincount(pb, minlength=B).max()), 3), 1e3, dtype=torch.float64)  # padding far away
    pts[pb, pr] = flat[pb, pq]
    pidx = torch.zeros(pts.shape[:2], dtype=torch.long).index_put((pb, pr), pq)
    coords = region.nonzero()
    rb = coords[:, 0]
    rr = per_case_rank(rb, B)
    Vr = int(torch.bincount(rb, minlength=B).max())
    vox = torch.full((B, Vr, 3), -1e3, dtype=torch.float64)
    vox[rb, rr] = coords[:, 1:].double() * T.H
    if near is None:  # the voxel → centreline-point assignment does not depend on θ: it can be reused
        exact = "donot_use_mm_for_euclid_dist"  # exact distances: the same nearest point for B = 1 and B = 6
        near = torch.cat([torch.cdist(vox[:, i : i + chunk], pts, compute_mode=exact).argmin(2)
                          for i in range(0, Vr, chunk)], 1)  # fmt: skip
        near = pidx.gather(1, near)  # (B, Vr) index into the E·Pm points
    s = torch.zeros(B, Vr, dtype=soft.dtype).index_put((rb, rr), soft[region])  # soft occupancy per region voxel
    vol_pt = torch.zeros(B, E * Pm, dtype=soft.dtype).scatter_add(1, near, s * T.H**3)  # (B, E·Pm)
    vol = vol_pt.reshape(B, E, Pm).sum(2)
    L = g["length"]
    r = torch.where(g["edge_valid"], (vol / (math.pi * L.clamp(min=1e-9))).clamp(min=0).sqrt(), torch.zeros_like(vol))
    # radius profile and stenosis (detached: a report flag)
    okp = valid.reshape(B, E, Pm)
    ds = (L / (okp.sum(2) - 1).clamp(min=1)).detach()
    rp = (vol_pt.detach().reshape(B, E, Pm) / (math.pi * ds[..., None].clamp(min=1e-9))).sqrt()
    k = torch.ones(1, 1, 9, dtype=torch.float64)
    num = F.conv1d(torch.where(okp, rp, 0.0).reshape(-1, 1, Pm), k, padding=4).reshape(B, E, Pm)
    den = F.conv1d(okp.double().reshape(-1, 1, Pm), k, padding=4).reshape(B, E, Pm)
    rs = num / den.clamp(min=1)
    n = okp.sum(2, keepdim=True)
    i = torch.arange(Pm)[None, None, :]
    inner = okp & (i >= 4) & (i < n - 4)
    rmin = torch.where(inner, rs, math.inf).min(2).values
    rmed = torch.where(inner, rs, math.nan).nanmedian(2).values
    severity = torch.where(inner.sum(2) >= 4, 1 - rmin / rmed, torch.zeros_like(rmin)).nan_to_num(0.0)
    # mean radius per vessel label (length-weighted) and the right/left asymmetry 2(r_R − r_L)/(r_R + r_L)
    onehot = F.one_hot(g["edge_label"], 16).double() * g["edge_valid"][..., None]
    rlab = torch.einsum("bel,be->bl", onehot, r * L) / torch.einsum("bel,be->bl", onehot, L).clamp(min=1e-12)
    present = torch.einsum("bel->bl", onehot) > 0
    asym = {}
    for name, (right, left) in T.PAIRS.items():
        both = present[:, right] & present[:, left]
        ai = 2 * (rlab[:, right] - rlab[:, left]) / (rlab[:, right] + rlab[:, left]).clamp(min=1e-12)
        asym[name] = torch.where(both, ai, torch.tensor(math.nan, dtype=torch.float64))
    return {"near": near, "dice": seg["dice"], "radius": r, "volume": vol, "length": L, "profile": rs,
            "severity": severity, "stenosis": severity >= 0.5,
            "r_label": rlab, "present": present, "asym": asym}  # fmt: skip


# ---------------------------------------------------------------- stage 4: boundaries
def boundaries(g: dict) -> dict:
    """Inlets at the free ends of the ICAs and the basilar artery, outlets at the free ends of the MCAs, ACAs and
    PCAs (Pcom and Acom ends are never boundaries). Each boundary gets its terminal edge and an outward normal."""
    kind = g["node_kind"]
    inlet = torch.isin(kind, torch.tensor(T.INLETS))
    outlet = torch.isin(kind, torch.tensor(T.OUTLETS))
    B, E, Pm, _ = g["poly"].shape
    u = g["edges"][..., 0]
    is_end = (kind.gather(1, u) > 0) & g["edge_valid"]  # edges that start at a free end (the first of a chain)
    bedge = torch.full(kind.shape, -1, dtype=torch.long)
    b_idx, e_idx = is_end.nonzero(as_tuple=True)
    bedge[b_idx, u[b_idx, e_idx]] = e_idx
    n_pts = (~torch.isnan(g["poly"][..., 0])).sum(2)
    k_in = (n_pts.clamp(min=2) - 1).clamp(max=4)  # a point ~2 mm into the vessel
    inner = g["poly"].gather(2, k_in[..., None, None].expand(B, E, 1, 3))[:, :, 0]
    direction = g["poly"][:, :, 0] - inner
    direction = direction / direction.norm(dim=2, keepdim=True).clamp(min=1e-12)
    normal = torch.zeros(B, kind.shape[1], 3, dtype=torch.float64)
    normal[b_idx, u[b_idx, e_idx]] = direction[b_idx, e_idx]
    return {"inlet": inlet, "outlet": outlet, "edge": bedge, "normal": normal}


# ---------------------------------------------------------------- stage 5: flow
def flow(m: dict, bnd: dict, g: dict, occluded: torch.Tensor, severity: torch.Tensor | None = None) -> dict:
    """Steady Poiseuille flow in the vessel network: conductance G_e per edge, Murray loads g_k to 0 Pa at the
    outlets, P_in at every open inlet (the left ICA inlet closed where `occluded`), one batched solve K p = b.

    `severity` (B,) in [0, 1] generalises `occluded`: the left-ICA inlet is fed through a stenosis of length
    STENOSIS_LENGTH whose lumen diameter is (1 − s) of the inlet vessel's, in series (Poiseuille: conductance
    G_s = G_ref a/(1 − a), a = (1 − s)⁴, so the inlet's total conductance drops as a). s = 0 is the open Dirichlet
    inlet and s = 1 the closed one, exactly the two workflow conditions."""
    G = torch.where(g["edge_valid"], eq.poiseuille_conductance(m["radius"], m["length"], T.MU), 0.0)
    B, N = g["node_kind"].shape
    r_out = m["radius"].gather(1, bnd["edge"].clamp(min=0)) * (bnd["edge"] >= 0)
    g_out = eq.murray_conductance(r_out, bnd["outlet"] & (bnd["edge"] >= 0), T.Q_TOT, T.P_IN)
    s = (occluded.to(G.dtype) if severity is None else severity.to(G.dtype))[:, None]
    lica = bnd["inlet"] & (g["node_kind"] == T.L_ICA)
    inlet = bnd["inlet"] & ~(lica & (s > 0))  # Dirichlet inlets
    a = (1 - s).clamp(min=0) ** 4
    stenosed = lica & (s > 0) & (s < 1)
    r_in = m["radius"].gather(1, bnd["edge"].clamp(min=0))  # radius of the vessel at each free end
    g_ref = eq.poiseuille_conductance(r_in, torch.full_like(r_in, T.STENOSIS_LENGTH), T.MU)
    g_s = torch.where(stenosed, g_ref * a / (1 - a).clamp(min=1e-12), 0.0)  # inlet fed through the stenosis
    leak = 1e-9 * T.Q_TOT / T.P_IN  # a negligible leak to 0 Pa: keeps a node cut off from every boundary at p = 0
    K = eq.kirchhoff_matrix(G, g["edges"], N) + torch.diag_embed(g_out + leak + g_s)
    eye = torch.eye(N, dtype=K.dtype).expand(B, N, N)
    K = torch.where(inlet[..., None], eye, K)  # Dirichlet rows p = P_in
    p = torch.linalg.solve(K, torch.where(inlet, T.P_IN, g_s * T.P_IN).to(K.dtype))
    u, v = g["edges"][..., 0], g["edges"][..., 1]
    q = G * (p.gather(1, u) - p.gather(1, v))  # (B, E) flow u → v (m³/s)
    net_out = torch.zeros_like(p).scatter_add(1, u, q).scatter_add(1, v, -q)  # flow leaving each node via edges
    q_in = torch.where(inlet, net_out, torch.where(stenosed, g_s * (T.P_IN - p), 0.0))
    return {"p": p, "q": q, "Q_out": g_out * p, "Q_in": q_in, "G": G, "inlet": inlet | stenosed}


def stenosis_sweep(m: dict, bnd: dict, g: dict, steps: int = T.SWEEP_STEPS) -> tuple[torch.Tensor, list[dict]]:
    """Left-ICA severity s = 0 … 1 in `steps` steps: one batched flow solve over all cases per step."""
    sev = torch.linspace(0, 1, steps, dtype=torch.float64)
    B = g["node_kind"].shape[0]
    return sev, [flow(m, bnd, g, torch.zeros(B, dtype=torch.bool), torch.full((B,), float(s))) for s in sev]


# ---------------------------------------------------------------- stage 6: report, stage 7: cohort
COLUMNS = ["case", "condition", "dice", "nodes", "edges", "length_mm", "stenoses"] + [
    f"r_{n}_mm" for n in ("BA", "R-ICA", "L-ICA", "R-MCA", "L-MCA", "R-ACA", "L-ACA", "R-PCA", "L-PCA", "R-Pcom",
                         "L-Pcom", "Acom")] + [f"AI_{n}" for n in T.PAIRS] + [
    "Q_in_R-ICA", "Q_in_L-ICA", "Q_in_BA", "Q_R-MCA", "Q_L-MCA", "Q_R-ACA", "Q_L-ACA", "Q_R-PCA", "Q_L-PCA",
    "Q_out_total", "share_L-MCA", "dp_max_mmHg"]  # fmt: skip
NAME_TO_LABEL = {v: k for k, v in T.LABELS.items()}


def report(m: dict, bnd: dict, g: dict, fl: dict) -> dict:
    """The numbers of one (case, condition): (B,) tensors keyed by column (flows in ml/s, radii in mm)."""
    kind = g["node_kind"]
    out = {"dice": m["dice"], "nodes": g["node_valid"].sum(1).double(), "edges": g["edge_valid"].sum(1).double(),
           "length_mm": g["length"].sum(1) * 1e3,
           "stenoses": (m["stenosis"] & g["edge_valid"]).sum(1).double()}  # fmt: skip
    for n in ("BA", "R-ICA", "L-ICA", "R-MCA", "L-MCA", "R-ACA", "L-ACA", "R-PCA", "L-PCA", "R-Pcom", "L-Pcom", "Acom"):
        lab = NAME_TO_LABEL[n]
        out[f"r_{n}_mm"] = torch.where(m["present"][:, lab], m["r_label"][:, lab] * 1e3, math.nan)
    for n in T.PAIRS:
        out[f"AI_{n}"] = m["asym"][n]

    def at(values, label):  # sum of a node quantity over the free ends with this label
        return torch.where(kind == label, values, 0.0).sum(1) * 1e6

    for n in ("R-ICA", "L-ICA", "BA"):
        out[f"Q_in_{n}"] = at(fl["Q_in"], NAME_TO_LABEL[n])
    for n in ("R-MCA", "L-MCA", "R-ACA", "L-ACA", "R-PCA", "L-PCA"):
        out[f"Q_{n}"] = at(fl["Q_out"], NAME_TO_LABEL[n])
    out["Q_out_total"] = fl["Q_out"].sum(1) * 1e6
    out["share_L-MCA"] = out["Q_L-MCA"] / out["Q_out_total"].clamp(min=1e-12)
    p_out = torch.where(bnd["outlet"], fl["p"], math.inf).min(1).values
    out["dp_max_mmHg"] = (T.P_IN - p_out) / T.MMHG
    return out


def cohort_csv(rows: list[dict]) -> str:
    """The cohort table (one row per case and condition) as CSV text, numbers to 9 significant digits."""
    lines = [",".join(COLUMNS)]
    for r in rows:
        lines.append(",".join(r[c] if c in ("case", "condition") else f"{float(r[c]):.9g}" for c in COLUMNS))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the tensor workflow engine
def parse_snakefile(text: str) -> list[dict]:
    """Rules of the Snakefile in order: name, output patterns, and inputs as (pattern, aggregated wildcards) where
    expand(pattern, case=…, …) aggregates over the named wildcards."""
    rules = []
    for block in re.split(r"^rule ", text, flags=re.M)[1:]:
        name = block.split(":", 1)[0].strip()
        sec = dict(re.findall(r"^    (input|output|log|shell):\s*(.*?)(?=^    \w+:|^\S|\Z)", block, flags=re.M | re.S))
        ins = []
        for m in re.finditer(r'expand\(\s*"([^"]+)"([^)]*)\)|"([^"]+)"', sec.get("input", "")):
            if m.group(1):
                ins.append((m.group(1), tuple(re.findall(r"(\w+)\s*=", m.group(2)))))
            else:
                ins.append((m.group(3), ()))
        outs = re.findall(r'"([^"]+)"', sec.get("output", ""))
        rules.append({"name": name, "inputs": ins, "outputs": outs})
    return rules


def rule_graph(rules: list[dict]) -> tuple[torch.Tensor, list[tuple]]:
    """A_rule (R, R): (A_rule)_{sr} = 1 when rule s reads a file pattern rule r writes; and each rule's wildcards
    (those of its outputs, or of its inputs for a target rule without outputs)."""
    R = len(rules)
    A = torch.zeros(R, R)
    for s, rs in enumerate(rules):
        for pat, _ in rs["inputs"]:
            for r, rr in enumerate(rules):
                if pat in rr["outputs"]:
                    A[s, r] = 1
    axes = []
    for r in rules:
        pats = r["outputs"] or [p for p, agg in r["inputs"]]
        wc = {w for p in pats for w in re.findall(r"\{(\w+)\}", p)}
        wc -= {w for p, agg in r["inputs"] for w in agg} if not r["outputs"] else set()
        axes.append(tuple(a for a in ("case", "condition") if a in wc))
    return A, axes


def job_list(rules: list[dict], axes: list, sizes: dict, values: dict) -> list[tuple]:
    """Jobs as (rule, case, condition), numbered rule by rule and row-major over the wildcard axes."""
    jobs = []
    for r, ax in zip(rules, axes):
        grids = [values[a] if a in ax else ("",) for a in sizes]
        for case in grids[0]:
            for cond in grids[1]:
                jobs.append((r["name"], case, cond))
    return jobs


def build_dag(text: str, cases=T.CASES, conditions=T.CONDITIONS) -> dict:
    """Everything the engine knows about the workflow: rules, A_rule, job list, job adjacency A (Kronecker), closure."""
    rules = parse_snakefile(text)
    A_rule, axes = rule_graph(rules)
    sizes = {"case": len(cases), "condition": len(conditions)}
    A, off = eq.job_adjacency(A_rule, axes, sizes)
    jobs = job_list(rules, axes, sizes, {"case": cases, "condition": conditions})
    return {"rules": rules, "A_rule": A_rule, "axes": axes, "A": A, "off": off, "jobs": jobs, "sizes": sizes}


def stage_functions(theta: torch.Tensor) -> dict:
    """What each rule does, as one batched call over all its jobs. `store` holds every finished rule's outputs.
    Case-axis stages see B cases; the (case, condition) stages see B·C rows, case-major like the job numbering."""

    def seg(st):
        return segment(st["image"], st["labels"], theta)

    def rep(x, C):  # broadcast case-level tensors to (case, condition) rows: the I ⊗ 𝟙 edge
        return {k: rep(v, C) if isinstance(v, dict) else v.repeat_interleave(C, 0) for k, v in x.items()}

    def flw(st):
        C = len(st["conditions"])
        occl = torch.tensor([c == "lica_occluded" for c in st["conditions"]] * len(st["cases"]))
        g, m, b = rep(st["centreline"], C), rep(st["measure"], C), rep(st["boundaries"], C)
        return {"flow": flow(m, b, g, occl), "g": g, "b": b}

    def rpt(st):
        C = len(st["conditions"])
        f = st["flow"]
        return report(rep(st["measure"], C), f["b"], f["g"], f["flow"])

    def coh(st):
        r = st["report"]
        rows = [{"case": c, "condition": k} for c in st["cases"] for k in st["conditions"]]
        for i, row in enumerate(rows):
            row.update({col: r[col][i] for col in r})
        return cohort_csv(rows)

    return {"segment": seg, "centreline": lambda st: centreline(st["labels"]),
            "measure": lambda st: measure(st["segment"], st["centreline"]),
            "boundaries": lambda st: boundaries(st["centreline"]), "flow": flw, "report": rpt, "cohort": coh,
            "all": lambda st: None}  # fmt: skip


def run_engine(dag: dict, cases=T.CASES, conditions=T.CONDITIONS, theta: float = T.THETA, grad: bool = False):
    """Run the DAG as tensors: rules level by level (levels of A_rule), each rule ONE batched call over its jobs.
    Returns the store and the wall-clock (start, end) of each rule call, relative to the start (s)."""
    t0 = time.perf_counter()
    image, labels = T.load_cases(cases)
    th = torch.full((len(cases),), theta, dtype=torch.float64, requires_grad=grad)
    store = {"image": image, "labels": labels, "cases": cases, "conditions": conditions, "theta": th}
    times = {"load": (0.0, time.perf_counter() - t0)}
    funcs = stage_functions(th)
    lev = eq.levels(dag["A_rule"])
    for k in range(int(lev.max()) + 1):
        for r in (lev == k).nonzero().flatten().tolist():
            name = dag["rules"][r]["name"]
            a = time.perf_counter() - t0
            with torch.set_grad_enabled(grad):
                store[name] = funcs[name](store)
            times[name] = (a, time.perf_counter() - t0)
    return store, times


# ---------------------------------------------------------------- the 3D scene: iso-surface of the segmentation
CUBE = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
TETS = torch.tensor([[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]])


def iso_surface(field: torch.Tensor, level: float, h: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Marching tetrahedra (the table-free member of the marching-cubes family): every cube that straddles the level
    is cut into 6 tetrahedra; a tetrahedron with 1 or 3 corners inside gives a triangle, with 2 a quad (2 triangles).
    Vertices sit on the cut grid edges (linear interpolation) and are shared, so the surface is watertight.
    field (X, Y, Z), inside where field > level → vertices (V, 3) in m and triangles (F, 3), normals outward."""
    X, Y, Z = field.shape
    inside = field > level
    c = torch.stack(torch.meshgrid(torch.arange(X - 1), torch.arange(Y - 1), torch.arange(Z - 1), indexing="ij"), -1)
    c = c.reshape(-1, 3)
    corner = c[:, None, :] + CUBE[None]  # (C, 8, 3)
    flat = (corner[..., 0] * Y + corner[..., 1]) * Z + corner[..., 2]  # (C, 8) global grid index
    ins = inside.flatten()[flat]
    mixed = ins.any(1) & ~ins.all(1)
    tet = flat[mixed][:, TETS].reshape(-1, 4)  # (Nt, 4)
    tin = inside.flatten()[tet]
    n_in = tin.sum(1)
    tet, tin, n_in = tet[(n_in > 0) & (n_in < 4)], tin[(n_in > 0) & (n_in < 4)], n_in[(n_in > 0) & (n_in < 4)]
    order = torch.argsort((~tin).long(), dim=1, stable=True)  # inside corners first
    t = tet.gather(1, order)
    one = n_in == 1
    three = n_in == 3
    two = n_in == 2
    tri_edges = []  # each triangle as 3 grid edges (a, b), a inside, b outside
    a1 = t[one]
    tri_edges.append(torch.stack([torch.stack([a1[:, 0], a1[:, k]], 1) for k in (1, 2, 3)], 1))
    a3 = t[three]
    tri_edges.append(torch.stack([torch.stack([a3[:, k], a3[:, 3]], 1) for k in (0, 1, 2)], 1))
    a2 = t[two]
    ac, ad, bd, bc = (torch.stack([a2[:, i], a2[:, j]], 1) for i, j in ((0, 2), (0, 3), (1, 3), (1, 2)))
    tri_edges += [torch.stack([ac, ad, bd], 1), torch.stack([ac, bd, bc], 1)]
    te = torch.cat(tri_edges)  # (F, 3, 2)
    key = te.min(2).values * (X * Y * Z) + te.max(2).values
    ukey, inv = torch.unique(key.flatten(), return_inverse=True)
    ea, eb = ukey // (X * Y * Z), ukey % (X * Y * Z)
    f = field.flatten().double()
    grid = torch.stack(torch.meshgrid(torch.arange(X), torch.arange(Y), torch.arange(Z), indexing="ij"), -1)
    grid = grid.reshape(-1, 3).double()
    s = ((level - f[ea]) / (f[eb] - f[ea])).clamp(0, 1)[:, None]
    verts = (grid[ea] + s * (grid[eb] - grid[ea])) * h
    tris = inv.reshape(-1, 3)
    # orient outward: the normal must point from the inside corner to the outside one
    a, b, cc = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    nrm = torch.cross(b - a, cc - a, dim=1)
    out_dir = grid[te[:, 0, 1]] - grid[te[:, 0, 0]]
    flip = torch.einsum("fi,fi->f", nrm, out_dir) < 0
    tris = torch.where(flip[:, None], tris[:, [0, 2, 1]], tris)
    area = nrm.norm(dim=1)
    return verts, tris[area > 1e-30 * h * h]
