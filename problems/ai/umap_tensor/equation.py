"""UMAP as tensor operations, one function per relation of the index form.

Shapes: N points, D input dimensions, k neighbours (the point itself included, as in umap-learn), E graph edges,
2 output dimensions. Summed indices are named in each docstring.
"""

import math

import torch


def sq_dist(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """d²_ij = x_id x_id + z_jd z_jd - 2 x_id z_jd (repeated d ⇒ summed), (N, M), never below 0."""
    xx = torch.einsum("id,id->i", x, x)  # (N,)
    zz = torch.einsum("jd,jd->j", z, z)  # (M,)
    return (xx[:, None] + zz[None, :] - 2 * torch.einsum("id,jd->ij", x, z)).clamp(min=0)


def knn(x: torch.Tensor, k: int, block: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact k nearest neighbours by brute force: all distances, then topk. dist, idx (N, k), the point itself first.
    block > 0 computes the (block, N) rows at a time, so memory is O(block·N) instead of O(N²)."""
    n = len(x)
    rows = range(0, n, block or n)
    d, ix = [], []
    for r0 in rows:  # one pass when block is None
        d2 = sq_dist(x[r0 : r0 + (block or n)], x)  # (B, N)
        own = torch.arange(r0, min(r0 + (block or n), n), device=x.device)
        d2[torch.arange(len(own), device=x.device), own] = -1.0  # the point itself always ranks first
        v, i = torch.topk(d2, k, dim=1, largest=False)  # (B, k), sorted ascending
        d.append(v.clamp(min=0).sqrt())
        ix.append(i)
    return torch.cat(d), torch.cat(ix)


def smooth_knn(dist: torch.Tensor, k: int, n_iter: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """ρ_i = distance to the nearest other point; σ_i solves Σ_j exp(-(d_ij - ρ_i)/σ_i) = log2 k (sum over the k-1
    neighbours j ≠ i) by bisection on ALL points at once: one (N,) tensor of brackets, 64 halvings.
    Like umap-learn, σ_i is kept above 10⁻³ × the mean neighbour distance of i (never active on these data)."""
    d = dist[:, 1:]  # (N, k-1) neighbours without the point itself
    rho = torch.where(d > 0, d, torch.full_like(d, math.inf)).min(1).values  # (N,) first non-zero distance
    rho = torch.where(torch.isinf(rho), torch.zeros_like(rho), rho)
    target = math.log2(k)
    lo = torch.zeros_like(rho)
    hi = torch.full_like(rho, math.inf)
    sigma = torch.ones_like(rho)
    gap = (d - rho[:, None]).clamp(min=0)  # (N, k-1)
    for _ in range(n_iter):
        s = torch.exp(-gap / sigma[:, None]).sum(1)  # (N,) the sum for the current guesses
        too_big = s > target  # σ too large → shrink
        hi = torch.where(too_big, sigma, hi)
        lo = torch.where(too_big, lo, sigma)
        sigma = torch.where(torch.isinf(hi), 2 * sigma, (lo + hi) / 2)
    return rho, torch.maximum(sigma, 1e-3 * d.mean(1))


def membership(dist: torch.Tensor, rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """p_{j|i} = exp(-(d_ij - ρ_i)/σ_i), (N, k); 1 for the nearest neighbour, 0 for the point itself."""
    p = torch.exp(-(dist - rho[:, None]).clamp(min=0) / sigma[:, None])
    p[:, 0] = 0.0
    return p


def fuzzy_union(p: torch.Tensor) -> torch.Tensor:
    """w_ij = p_ij + p_ji - p_ij p_ji: the probabilistic OR of the two directed memberships, (N, N) symmetric."""
    return p + p.T - p * p.T


def q(d2: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Low-dimensional similarity q_ij = 1 / (1 + a ‖y_i - y_j‖^{2b}), from the squared distance."""
    return 1.0 / (1.0 + a * d2.clamp(min=0) ** b)


def fit_ab(min_dist: float = 0.1, spread: float = 1.0, n_iter: int = 100) -> tuple[float, float]:
    """Least-squares fit of 1/(1 + a x^{2b}) to the target f(x) = 1 for x < min_dist, exp(-(x - min_dist)/spread)
    beyond, on 300 points in [0, 3·spread], by Gauss–Newton with Levenberg damping."""
    x = torch.linspace(0, 3 * spread, 300, dtype=torch.float64)[1:]  # x = 0 contributes 0 residual for any a, b
    f = torch.where(x < min_dist, torch.ones_like(x), torch.exp(-(x - min_dist) / spread))
    logx = torch.log(x)

    def residual(th):  # th = (a, b) → r (299,) and its Jacobian J (299, 2)
        u = th[0] * torch.exp(2 * th[1] * logx)  # a x^{2b}
        r = 1.0 / (1.0 + u) - f
        J = torch.stack([-u / th[0], -2 * logx * u], dim=1) / (1.0 + u)[:, None] ** 2
        return r, J

    th, lam = torch.tensor([1.0, 1.0], dtype=torch.float64), 1e-3
    for _ in range(n_iter):
        r, J = residual(th)
        step = torch.linalg.solve(J.T @ J + lam * torch.eye(2, dtype=th.dtype), -J.T @ r)
        new = th + step
        if (residual(new)[0] ** 2).sum() < (r**2).sum():
            th, lam = new, lam / 3
        else:
            lam *= 4
    return float(th[0]), float(th[1])


def forces(y, i, j, i_neg, k_neg, a, b, clip=4.0, eps=1e-3, w_att=None, w_rep=None):
    """Hand-written UMAP forces for the whole batch of sampled edges at once, (N, 2):
    attraction on edge (i, j):    f = -2ab d^{2(b-1)} / (1 + a d^{2b}) (y_i - y_j), added to i and subtracted from j;
    repulsion of negative (i, k): f = 2b / ((ε + d²)(1 + a d^{2b})) (y_i - y_k), added to i only.
    Each term is clipped to [-4, 4] per coordinate (umap-learn's clip); index_add_ sums the terms of a point."""
    out = torch.zeros_like(y)
    diff = y[i] - y[j]  # (E, 2)
    d2 = torch.einsum("ec,ec->e", diff, diff)  # (E,)
    c = torch.where(d2 > 0, -2 * a * b * d2.clamp(min=1e-30) ** (b - 1) / (1 + a * d2**b), torch.zeros_like(d2))
    g = c[:, None] * diff if w_att is None else (w_att * c)[:, None] * diff
    g = g if clip is None else g.clamp(-clip, clip)
    out.index_add_(0, i, g).index_add_(0, j, -g)
    diff = y[i_neg] - y[k_neg]  # (S, 2)
    d2 = torch.einsum("sc,sc->s", diff, diff)
    c = 2 * b / ((eps + d2) * (1 + a * d2**b))
    c = torch.where(i_neg == k_neg, torch.zeros_like(c), c)  # a point never repels itself
    g = c[:, None] * diff if w_rep is None else (w_rep * c)[:, None] * diff
    g = g if clip is None else g.clamp(-clip, clip)
    return out.index_add_(0, i_neg, g)


def sampled_loss(y, i, j, i_neg, k_neg, a, b, eps=0.0):
    """The cross-entropy UMAP samples: Σ_e -log q_ij (edges) + Σ_s -log(1 - q_ik) (negatives), with y_k detached so
    a negative sample moves only its head, as in umap-learn. ε shifts d² in the repulsive term (ε = 0: exact)."""
    d2 = ((y[i] - y[j]) ** 2).sum(1)  # (E,)
    att = torch.log1p(a * d2.clamp(min=1e-30) ** b)  # -log q
    d2n = ((y[i_neg] - y[k_neg].detach()) ** 2).sum(1) + eps  # (S,)
    keep = i_neg != k_neg
    u = a * d2n[keep] ** b
    rep = torch.log1p(u) - torch.log(u)  # -log(1 - q) = log((1 + u)/u)
    return att.sum() + rep.sum()


def cross_entropy(w: torch.Tensor, y: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Full fuzzy-set cross-entropy (without the constant entropy of w):
    C = Σ_{i≠j} -[w_ij log q_ij + (1 - w_ij) log(1 - q_ij)]  (double sum over i and j),
    computed as Σ_{i≠j} -log(1 - q_ij) over all pairs plus Σ_edges w_ij [log(1 - q_ij) - log q_ij] over the edges,
    with u = a d^{2b}: -log q = log(1 + u), -log(1 - q) = log(1 + 1/u)."""
    u = a * sq_dist(y, y).clamp(min=1e-12) ** b  # (N, N)
    rep = torch.log1p(1 / u)
    rep.fill_diagonal_(0.0)
    i, j = torch.nonzero(w, as_tuple=True)  # (E,) edges
    return rep.sum() + (w[i, j] * (torch.log1p(u[i, j]) - rep[i, j])).sum()


def _ranks(x: torch.Tensor) -> torch.Tensor:
    """r_ij = rank of j among i's neighbours by distance (1 = nearest), (N, N); argsort of argsort."""
    d2 = sq_dist(x, x)
    d2.fill_diagonal_(math.inf)
    return torch.argsort(torch.argsort(d2, dim=1), dim=1) + 1


def _knn_idx(x: torch.Tensor, k: int) -> torch.Tensor:
    return knn(x, k + 1)[1][:, 1:]  # (N, k) without the point itself


def trustworthiness(x: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """T(k) = 1 - 2/(N k (2N - 3k - 1)) Σ_i Σ_{j ∈ U_k(i)} (r_ij - k): U_k(i) are the k nearest neighbours of i in
    the embedding y that are not among its k nearest in x, r_ij their rank in x (Venna & Kaski 2001)."""
    n = len(x)
    r = _ranks(x.double()).gather(1, _knn_idx(y.double(), k))  # (N, k) input ranks of embedding neighbours
    return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * float((r - k).clamp(min=0).sum())


def continuity(x: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """C(k): trustworthiness with the roles swapped: input neighbours that the embedding pushed away."""
    return trustworthiness(y, x, k)


def knn_recall(x: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Fraction of the k nearest neighbours in x that are also among the k nearest in y."""
    n = len(x)
    in_x = torch.zeros(n, n, dtype=torch.bool, device=x.device)
    in_x.scatter_(1, _knn_idx(x.double(), k), True)
    return float(in_x.gather(1, _knn_idx(y.double(), k)).float().mean())


def knn_accuracy(y: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Leave-one-out k-NN classification in the embedding: each point takes the majority label of its k nearest
    other points (votes = Σ_j one_hot(label_j) over the neighbours)."""
    votes = torch.nn.functional.one_hot(labels[_knn_idx(y.double(), k)], int(labels.max()) + 1).sum(1)  # (N, C)
    return float((votes.argmax(1) == labels).float().mean())
