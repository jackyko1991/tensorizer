r"""Single-cell analysis → tensor translation.

Counts K_{cg}: cell c, gene g. The standard first steps of every single-cell RNA-seq analysis:

    n_c      = Σ_g K_{cg}                                   library size (g summed)
    X_{cg}   = log(1 + 10⁴ K_{cg} / n_c)                     depth normalisation + log
    z_g      = standardised log(Var_c X̃_{cg} / Mean_c X̃_{cg}) highly-variable-gene score (within bins of mean)
    Z_{cg}   = X_{cg} − ⟨X_{cg}⟩_c = U_{cp} S_p V_{gp}       centre, then SVD
    y_{cp}   = Z_{cg} V_{gp}                                 PC scores (g summed); explained variance S_p² / Σ S²

k-means in PC space, for R restarts r at once (p summed in the distances, c summed in the means):

    d_{rck}  = y_{cp}y_{cp} − 2 y_{cp}μ_{rkp} + μ_{rkp}μ_{rkp}
    A_{rck}  = 1 if k = argmin_k d_{rck}
    μ_{rkp}  = A_{rck} y_{cp} / A_{rck}
"""

import torch


def library_size(counts: torch.Tensor) -> torch.Tensor:
    """n_c = Σ_g K_{cg}.   (C, G) → (C,);  g summed."""
    return torch.einsum("cg->c", counts)


def normalise(counts: torch.Tensor, scale: float = 1e4) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth-normalised counts X̃ = scale · K_{cg} / n_c (counts per 10⁴, the CPM idea) and X = log(1 + X̃)."""
    scaled = scale * counts / library_size(counts)[:, None]
    return scaled, torch.log1p(scaled)


def hvg_scores(scaled: torch.Tensor, n_bins: int = 20) -> torch.Tensor:
    """Highly-variable-gene score z_g (G,): the dispersion D_g = Var/Mean of the normalised counts, standardised
    among genes of similar mean expression.

    Noise alone makes D_g grow with expression (negative-binomial variance m + φm²), so genes are put into
    n_bins bins of log mean, and z_g = (log D_g − bin mean) / bin std.  The bin averages are one-hot sums:
    B_{gb} = 1 if gene g is in bin b,  ⟨log D⟩_b = B_{gb} log D_g / B_{gb}  (g summed).
    """
    mean = scaled.mean(0).clamp_min(1e-8)
    log_d = torch.log(scaled.var(0) / mean + 1e-12)  # (G,)
    log_m = torch.log(mean)
    edges = torch.quantile(log_m, torch.linspace(0, 1, n_bins + 1, device=scaled.device)[1:-1])
    bin_of = torch.bucketize(log_m, edges)  # (G,) bin index
    B = torch.nn.functional.one_hot(bin_of, n_bins).float()  # (G, n_bins)
    count = torch.einsum("gb->b", B)
    bin_mean = torch.einsum("gb,g->b", B, log_d) / count
    bin_std = (torch.einsum("gb,g->b", B, log_d**2) / count - bin_mean**2).clamp_min(1e-12).sqrt()
    return (log_d - bin_mean[bin_of]) / bin_std[bin_of]


def pca(X: torch.Tensor, n_pcs: int):
    """Centre the cells × genes matrix and take its SVD Z = U S Vᵀ.

    Returns the scores y_{cp} = Z_{cg} V_{gp} (C, P), the loadings V (G, P) and the explained-variance ratio
    S_p² / Σ_q S_q² of every component (sorted from largest to smallest, sums to 1 over all components).
    """
    Z = X - X.mean(0)  # (C, G)
    U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    V = Vh[:n_pcs].T  # (G, P)
    scores = torch.einsum("cg,gp->cp", Z, V)  # = U[:, :P] * S[:P]
    return scores, V, S**2 / (S**2).sum()


def sq_distances(y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """d_{rck} = ‖y_c − μ_rk‖².   y: (C, P), mu: (R, K, P) → (R, C, K);  p summed."""
    yy = torch.einsum("cp,cp->c", y, y)
    ym = torch.einsum("cp,rkp->rck", y, mu)
    mm = torch.einsum("rkp,rkp->rk", mu, mu)
    return (yy[None, :, None] - 2 * ym + mm[:, None, :]).clamp_min(0.0)


def assign(d: torch.Tensor) -> torch.Tensor:
    """One-hot assignment A_{rck} to the nearest centroid.   (R, C, K) → (R, C, K)."""
    return torch.nn.functional.one_hot(d.argmin(-1), d.shape[-1]).float()


def update_centroids(y: torch.Tensor, A: torch.Tensor, mu_old: torch.Tensor) -> torch.Tensor:
    """μ_{rkp} = Σ_c A_{rck} y_{cp} / Σ_c A_{rck};  an empty cluster keeps its old centre.   → (R, K, P);  c summed."""
    total = torch.einsum("rck,cp->rkp", A, y)
    count = torch.einsum("rck->rk", A)[..., None]
    return torch.where(count > 0, total / count.clamp_min(1.0), mu_old)


def inertia(d: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Within-cluster sum of squares Σ_{c,k} A_{rck} d_{rck} for each restart.   → (R,)."""
    return torch.einsum("rck,rck->r", A, d)


def adjusted_rand_index(a: torch.Tensor, b: torch.Tensor) -> float:
    """ARI of two labelings (C,) via the contingency table n_{ij} = Σ_c [a_c = i][b_c = j]  (c summed).

    ARI = (Σ C(n_ij,2) − E) / (½(Σ C(a_i,2) + Σ C(b_j,2)) − E),  E = Σ C(a_i,2) Σ C(b_j,2) / C(n,2):
    1 for identical partitions (up to renaming), about 0 for random ones.
    """
    A = torch.nn.functional.one_hot(a).double()
    B = torch.nn.functional.one_hot(b).double()
    n = torch.einsum("ci,cj->ij", A, B)

    def pairs(x: torch.Tensor) -> torch.Tensor:
        return (x * (x - 1) / 2).sum()

    index, sum_a, sum_b = pairs(n), pairs(n.sum(1)), pairs(n.sum(0))
    expected = sum_a * sum_b / pairs(torch.tensor(float(len(a)), dtype=torch.float64))
    return ((index - expected) / (0.5 * (sum_a + sum_b) - expected)).item()


def tsne_affinities(y: torch.Tensor, perplexity: float, n_bisect: int = 40) -> torch.Tensor:
    """t-SNE input similarities P_{ij}: a Gaussian around each cell with its width tuned to the perplexity.

    p_{j|i} ∝ exp(−β_i ‖y_i − y_j‖²), with β_i found by bisection (all cells at once) so that the entropy is
    log(perplexity); then P = (p_{j|i} + p_{i|j}) / 2C.
    """
    C = y.shape[0]
    d = torch.cdist(y, y) ** 2  # (C, C)
    d.fill_diagonal_(float("inf"))
    lo = torch.full((C, 1), 1e-6, dtype=y.dtype, device=y.device)
    hi = torch.full((C, 1), 1e3, dtype=y.dtype, device=y.device)
    target = torch.log(torch.tensor(perplexity, dtype=y.dtype))
    for _ in range(n_bisect):
        beta = (lo * hi).sqrt()  # bisection on a log scale
        p = torch.softmax(-beta * d, dim=1)
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(1, keepdim=True)
        too_wide = entropy > target  # too many neighbours → sharpen (larger β)
        lo = torch.where(too_wide, beta, lo)
        hi = torch.where(too_wide, hi, beta)
    p = torch.softmax(-(lo * hi).sqrt() * d, dim=1)
    return (p + p.T) / (2 * C)


def tsne_loss(Y: torch.Tensor, P: torch.Tensor, exaggeration: float = 1.0) -> torch.Tensor:
    """KL(P ‖ Q) up to a constant, −Σ_ij P_ij log Q_ij, with Student-t similarities in the 2D map:

    q_ij = w_ij / Σ_{k≠l} w_kl,   w_ij = 1 / (1 + d_ij),   d_ij = Y_ia Y_ia + Y_ja Y_ja − 2 Y_ia Y_ja  (a summed).
    Since Σ P = 1:  −Σ P log Q = Σ P_ij log(1 + d_ij) + log Σ w.   Early on P is multiplied by an
    "exaggeration" factor, which pulls clusters together before they settle.
    """
    sq = torch.einsum("ia,ia->i", Y, Y)
    d = (sq[:, None] + sq[None, :] - 2 * torch.einsum("ia,ja->ij", Y, Y)).clamp_min(0.0)  # (C, C)
    w_sum = (1.0 / (1.0 + d)).sum() - len(Y)  # leave out the diagonal (d_ii = 0, w_ii = 1)
    return exaggeration * ((P * torch.log1p(d)).sum() + torch.log(w_sum))
