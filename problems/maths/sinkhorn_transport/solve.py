"""Sinkhorn in the log domain for a batch of ε, the Sinkhorn divergence, and an exact reference (Hungarian)."""

import numpy as np
import torch

from problems._common.progress import steps

from .equation import cost_matrix, plan, update_f, update_g


def sinkhorn(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: torch.Tensor, n_iter: int, eps_start: float | None = None
) -> dict:
    """Entropic OT for every ε_e at once. With eps_start, ε is annealed geometrically from eps_start to its target
    over the first half of the iterations (ε-scaling, warm-started potentials), which makes small ε converge.
    a (N,), b (M,), C (N, M), ε (E,) → plan (E, N, M), potentials f (E, N), g (E, M), cost ⟨P, C⟩ (E,),
    dual ⟨f, a⟩ + ⟨g, b⟩ = OT_ε (E,)."""
    log_a, log_b = a.log(), b.log()
    target = eps[:, None, None].to(c.dtype)  # (E, 1, 1)
    start = target.clamp(min=eps_start) if eps_start is not None else target
    g = torch.zeros(len(eps), c.shape[1], dtype=c.dtype, device=c.device)
    for t in steps(n_iter, "Sinkhorn iterations"):
        frac = min(1.0, 2 * t / n_iter) if eps_start is not None else 1.0
        e = start * (target / start) ** frac  # (E, 1, 1) current ε
        f = update_f(g, c, log_b, e)
        g = update_g(f, c, log_a, e)
    p = plan(f, g, c, log_a, log_b, target)
    return {
        "plan": p,
        "f": f,
        "g": g,
        "cost": torch.einsum("eij,ij->e", p, c),
        "dual": f @ a + g @ b,
    }


def sinkhorn_divergence(x, a, y, b, eps: torch.Tensor, n_iter: int) -> torch.Tensor:
    """S_ε = OT_ε(μ, ν) − ½ OT_ε(μ, μ) − ½ OT_ε(ν, ν) (Feydy et al. 2019): zero when μ = ν.   → (E,)"""
    xy = sinkhorn(a, b, cost_matrix(x, y), eps, n_iter)["dual"]
    xx = sinkhorn(a, a, cost_matrix(x, x), eps, n_iter)["dual"]
    yy = sinkhorn(b, b, cost_matrix(y, y), eps, n_iter)["dual"]
    return xy - 0.5 * xx - 0.5 * yy


def exact_assignment(c: torch.Tensor) -> torch.Tensor:
    """Exact optimal assignment (Hungarian algorithm with potentials, O(n³)): the slow, exact reference for ε → 0.
    C (N, M) with N ≤ M → column assigned to every row (N,)."""
    cost = c.detach().cpu().double().numpy()
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j] = row assigned to column j (1-based, 0 = free)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = cost[i0 - 1] - u[i0] - v[1:]  # reduced costs from row i0
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            j1 = int(np.flatnonzero(free)[np.argmin(minv[1:][free])]) + 1
            delta = minv[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:  # augment along the alternating path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    rows = np.zeros(n, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            rows[p[j] - 1] = j - 1
    return torch.from_numpy(rows).to(c.device)
