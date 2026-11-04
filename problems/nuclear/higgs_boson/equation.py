r"""Physics → tensor translation.

Diphoton events in a mass bin n are Poisson distributed around a smooth background plus a narrow Higgs peak:

    ν_n = w_n \exp(X_{nk} θ_k) + μ\, N_{SM}\, S_n(m_H, σ),          (k summed)
    S_n = Φ((m_{n+1} − m_H)/σ) − Φ((m_n − m_H)/σ)                   (Gaussian peak integrated over the bin)
    −\ln L = \sum_n (ν_n − k_n \ln ν_n)                              (Poisson likelihood, constants dropped)

The discovery test statistic is the profile likelihood ratio q₀ = −2 ln[L(μ = 0, \hat{\hat θ}) / L(\hat μ, \hat θ)]
for \hat μ ≥ 0 (else 0). By Wilks' theorem it follows ½δ(0) + ½χ²₁ without a signal, so Z = √q₀.

The Higgs field is a complex doublet; its neutral component φ sits in the Mexican-hat potential

    V(φ) = −μ²|φ|² + λ|φ|⁴,     minimum at |φ| = v/√2 with v = \sqrt{μ²/λ},     m_H² = 2λ v².
"""

import math

import torch


def background(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""B_n = \exp(X_{nk} θ_k) per GeV  (k summed).   theta: (..., K), x: (N, K) → (..., N)"""
    return torch.exp(torch.einsum("nk,...k->...n", x, theta))


def signal_shape(edges: torch.Tensor, m_h: torch.Tensor, sigma: float) -> torch.Tensor:
    r"""S_n: fraction of a Gaussian peak at m_H falling in bin n.   edges: (N+1,), m_h: (...) → (..., N)"""
    z = (edges - m_h[..., None]) / (sigma * math.sqrt(2))
    cdf = 0.5 * torch.erf(z)
    return cdf[..., 1:] - cdf[..., :-1]


def expected(theta, n_s, x, widths, shape) -> torch.Tensor:
    r"""ν_n = w_n B_n + n_s S_n: expected events per bin.   → (..., N)"""
    return widths * background(theta, x) + n_s[..., None] * shape


def nll(nu: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    r"""−ln L = Σ_n (ν_n − k_n ln ν_n) summed over the last axis.   → (...)"""
    return (nu - counts * torch.log(nu)).sum(-1)


def significance(q0: torch.Tensor) -> torch.Tensor:
    r"""Local significance Z = √q₀ (Wilks, one parameter of interest)."""
    return q0.clamp_min(0).sqrt()


def p_value(q0: torch.Tensor) -> torch.Tensor:
    r"""One-sided p₀ = 1 − Φ(√q₀) = ½ erfc(√(q₀/2))."""
    return 0.5 * torch.erfc((q0.clamp_min(0) / 2).sqrt())


def mexican_hat(phi: torch.Tensor, mu2: float, lam: float) -> torch.Tensor:
    r"""V(φ) = −μ²|φ|² + λ|φ|⁴ for a complex field φ (any shape)."""
    r2 = (phi * phi.conj()).real
    return -mu2 * r2 + lam * r2**2
