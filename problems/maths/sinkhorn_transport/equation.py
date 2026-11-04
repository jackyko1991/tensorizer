r"""Optimal transport → tensor translation.

Two discrete measures μ = Σ_i a_i δ_{x_i} and ν = Σ_j b_j δ_{y_j} and the cost C_ij = |x_i − y_j|². A coupling P ≥ 0
with row sums a and column sums b says how much mass moves from x_i to y_j; the squared Wasserstein-2 distance is

    W_2^2(\mu, \nu) = \min_{P\,\mathbf 1 = a,\ P^\top \mathbf 1 = b} \sum_{ij} P_{ij} C_{ij}.

Entropic OT (Cuturi 2013) adds ε KL(P | a ⊗ b). The optimum has the form P_ij = a_i b_j e^{(f_i + g_j − C_ij)/ε},
and Sinkhorn's alternating updates of the potentials f, g (here in the log domain, with logsumexp) converge to it:

    f_i = -\varepsilon \log \sum_j b_j\,e^{(g_j - C_{ij})/\varepsilon},\qquad
    g_j = -\varepsilon \log \sum_i a_i\,e^{(f_i - C_{ij})/\varepsilon}.

As ε → 0 the plan sharpens to the exact (unregularised) optimum. The Sinkhorn divergence
S_ε(μ, ν) = OT_ε(μ, ν) − ½ OT_ε(μ, μ) − ½ OT_ε(ν, ν) removes the entropic bias, so S_ε(μ, μ) = 0.

Closed forms used as checks: in 1D the optimal plan pairs sorted points (quantile functions); between Gaussians
W_2² = |m_1 − m_2|² + tr(S_1 + S_2 − 2 (S_1^{1/2} S_2 S_1^{1/2})^{1/2}) (the Bures metric).
"""

import torch


def cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """C_ij = |x_i − y_j|² = (x_ik − y_jk)(x_ik − y_jk)   (k summed).   (N, d), (M, d) → (N, M)"""
    diff = x[:, None, :] - y[None, :, :]
    return torch.einsum("ijk,ijk->ij", diff, diff)


def update_f(g: torch.Tensor, c: torch.Tensor, log_b: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""f_i = −ε logsumexp_j(log b_j + (g_j − C_ij)/ε)   (j summed).   (E, M), (N, M), (M,), (E, 1, 1) → (E, N)"""
    return -eps[:, :, 0] * torch.logsumexp(log_b + (g[:, None, :] - c) / eps, dim=-1)


def update_g(f: torch.Tensor, c: torch.Tensor, log_a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""g_j = −ε logsumexp_i(log a_i + (f_i − C_ij)/ε)   (i summed).   (E, N), (N, M), (N,), (E, 1, 1) → (E, M)"""
    return -eps[:, :, 0] * torch.logsumexp(log_a[:, None] + (f[:, :, None] - c) / eps, dim=-2)


def plan(f: torch.Tensor, g: torch.Tensor, c: torch.Tensor, log_a, log_b, eps: torch.Tensor) -> torch.Tensor:
    """P_ij = a_i b_j exp((f_i + g_j − C_ij)/ε).   (E, N), (E, M), (N, M) → (E, N, M)"""
    return torch.exp(log_a[:, None] + log_b + (f[:, :, None] + g[:, None, :] - c) / eps)


def barycentric_map(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """T(x_i) = Σ_j P_ij y_j / Σ_j P_ij: where the mass of x_i goes on average.   (N, M), (M, d) → (N, d)"""
    return torch.einsum("ij,jk->ik", p, y) / p.sum(-1, keepdim=True)


def interpolate(x: torch.Tensor, tx: torch.Tensor, t: float) -> torch.Tensor:
    """McCann's displacement interpolation: every particle moves on a straight line, x_t = (1 − t) x + t T(x)."""
    return (1 - t) * x + t * tx


def w2_squared_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1D closed form for equal-size uniform samples: mean (x_(i) − y_(i))² of the sorted values.   (N,), (N,) → ()"""
    return (torch.sort(x).values - torch.sort(y).values).pow(2).mean()


def sqrtm(s: torch.Tensor) -> torch.Tensor:
    """Symmetric positive square root through the eigendecomposition S = U diag(λ) Uᵀ.   (d, d) → (d, d)"""
    lam, u = torch.linalg.eigh(s)
    return u @ torch.diag(lam.clamp(min=0).sqrt()) @ u.T


def gaussian_w2_squared(m1, s1, m2, s2) -> torch.Tensor:
    """W_2² between N(m1, S1) and N(m2, S2): |m1 − m2|² + tr(S1 + S2 − 2 (S1^{½} S2 S1^{½})^{½}).   → ()"""
    r1 = sqrtm(s1)
    return (m1 - m2).pow(2).sum() + torch.trace(s1 + s2 - 2 * sqrtm(r1 @ s2 @ r1))


def gaussian_monge_matrix(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """A = S1^{−½} (S1^{½} S2 S1^{½})^{½} S1^{−½}: the optimal map between centred Gaussians is x ↦ A x.   (d, d)"""
    r1 = sqrtm(s1)
    r1_inv = torch.linalg.inv(r1)
    return r1_inv @ sqrtm(r1 @ s2 @ r1) @ r1_inv
