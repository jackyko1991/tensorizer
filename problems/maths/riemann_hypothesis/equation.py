r"""Mathematics → tensor translation.

1. Riemann–Siegel formula for the real function Z(t) = e^{iθ(t)} ζ(½ + it), vectorised over many heights t:

       Z(t) = 2 Σ_{n ≤ N} n^{−1/2} cos(θ(t) − t ln n) + (−1)^{N−1} a^{−1/2} [C₀(p) + C₁(p) a^{−1} + C₂(p) a^{−2}]

   with a = √(t/2π), N = ⌊a⌋, p = a − N and, from Ψ(p) = cos 2π(p² − p − 1/16) / cos 2πp,
   C₀ = Ψ,  C₁ = −Ψ⁽³⁾/(96π²),  C₂ = Ψ⁽²⁾/(64π²) + Ψ⁽⁶⁾/(18432π⁴).
   Ψ is entire; it and its derivatives come from one Chebyshev series. The sum over n is an einsum over the
   masked (T, N) matrix.

2. Zero counting. N(t) = #{zeros 0 < γ ≤ t} = θ(t)/π + 1 + S(t) (Riemann–von Mangoldt), with S(t) small.
   Gram points g_n solve θ(g_n) = nπ; "Gram's law" (−1)^n Z(g_n) > 0 usually holds.

3. Unfolding x = θ(γ)/π + 1 makes the mean spacing 1; Montgomery's pair correlation conjecture says the density of
   differences x_j − x_k is R₂(u) = 1 − (sin πu / πu)², the same as for eigenvalues of random Hermitian (GUE) matrices.

4. Explicit formula (von Mangoldt): ψ(x) = Σ_{p^k ≤ x} ln p = x − Σ_ρ x^ρ/ρ − ln 2π − ½ ln(1 − x^{−2}),
   the zeros taken in conjugate pairs: Σ_ρ x^ρ/ρ = Σ_k 2 Re(x^{ρ_k}/ρ_k).
"""

import functools
import math

import torch


def theta(t: torch.Tensor) -> torch.Tensor:
    r"""θ(t) = t/2 ln(t/2π) − t/2 − π/8 + 1/(48t) + 7/(5760t³) + 31/(80640t⁵)   (asymptotic, fine for t > 10)."""
    return (
        t / 2 * torch.log(t / (2 * math.pi))
        - t / 2
        - math.pi / 8
        + 1 / (48 * t)
        + 7 / (5760 * t**3)
        + 31 / (80640 * t**5)
    )


def psi(p: torch.Tensor) -> torch.Tensor:
    r"""Ψ(p) = cos 2π(p² − p − 1/16) / cos 2πp. Every zero of the denominator is also a zero of the numerator, so
    Ψ is entire, but the quotient loses precision near p = ¼, ¾, and its derivatives even more."""
    return torch.cos(2 * math.pi * (p * p - p - 1 / 16)) / torch.cos(2 * math.pi * p)


def chebyshev_basis(p: torch.Tensor, degree: int) -> torch.Tensor:
    r"""T_k(2p − 1) for k = 0 … degree by the recurrence T_{k+1} = 2z T_k − T_{k−1}.   (P,) → (P, degree + 1)"""
    z = 2 * p - 1
    basis = [torch.ones_like(z), z]
    for _ in range(degree - 1):
        basis.append(2 * z * basis[-1] - basis[-2])
    return torch.stack(basis, dim=-1)


@functools.cache
def psi_coefficients(degree: int = 26, device: torch.device | None = None) -> torch.Tensor:
    r"""Least-squares Chebyshev coefficients c_k of Ψ on [0, 1], fitted at 200 Chebyshev nodes (none of which lies
    within 1e-4 of ¼ or ¾, where the quotient is still accurate to ~1e-12). Computed once.   → (degree + 1,)"""
    j = torch.arange(200, dtype=torch.float64, device=device)
    nodes = 0.5 * (1 + torch.cos(math.pi * (j + 0.5) / 200))
    basis = chebyshev_basis(nodes, degree)  # (200, degree + 1)
    return torch.linalg.lstsq(basis, psi(nodes)[:, None]).solution[:, 0]


@functools.cache
def psi_derivative_coefficients(max_order: int = 6, device: torch.device | None = None) -> torch.Tensor:
    r"""Chebyshev coefficients of Ψ, Ψ′, …, Ψ⁽⁶⁾: the derivative of Σ c_k T_k(2p − 1) is again a Chebyshev series,
    with c′_{k−1} = c′_{k+1} + 2k c_k (times 2 from dz/dp = 2; c′_0 is halved).   → (max_order + 1, D + 1)"""
    rows = [psi_coefficients(device=device)]
    for _ in range(max_order):
        c, d = rows[-1], torch.zeros_like(rows[-1])
        for k in range(len(c) - 1, 0, -1):
            d[k - 1] = (d[k + 1] if k + 1 < len(c) else 0.0) + 2 * k * c[k]
        d[0] /= 2
        rows.append(2 * d)
    return torch.stack(rows)


def psi_derivatives(p: torch.Tensor) -> torch.Tensor:
    r"""[Ψ, Ψ′, …, Ψ⁽⁶⁾] at every p, from the differentiated Chebyshev series (smooth everywhere, no 0/0).
    (P,) → (7, P):  Ψ⁽ʲ⁾_p = B_{pk} c⁽ʲ⁾_k, k summed"""
    coeffs = psi_derivative_coefficients(device=p.device)  # (7, D + 1)
    basis = chebyshev_basis(p, coeffs.shape[1] - 1)  # (P, D + 1)
    return torch.einsum("pk,jk->jp", basis, coeffs)


def riemann_siegel_z(t: torch.Tensor) -> torch.Tensor:
    r"""Z(t) by the Riemann–Siegel formula with the corrections C₀, C₁, C₂.   t: (T,) float64 → (T,)"""
    a = torch.sqrt(t / (2 * math.pi))
    big_n = a.floor()
    n = torch.arange(1, int(big_n.max()) + 1, dtype=t.dtype, device=t.device)  # (N_max,)
    keep = n[None, :] <= big_n[:, None]  # (T, N_max): only n ≤ N(t) contributes
    phase = theta(t)[:, None] - t[:, None] * torch.log(n)[None, :]  # (T, N_max)
    terms = torch.where(keep, torch.cos(phase), 0.0)
    main = 2 * torch.einsum("tn,n->t", terms, n.rsqrt())  # repeated n ⇒ summed
    d = psi_derivatives(a - big_n)
    c0 = d[0]
    c1 = -d[3] / (96 * math.pi**2)
    c2 = d[2] / (64 * math.pi**2) + d[6] / (18432 * math.pi**4)
    sign = 1 - 2 * ((big_n - 1) % 2)  # (−1)^{N−1}
    return main + sign * a.rsqrt() * (c0 + c1 / a + c2 / a**2)


def smooth_count(t: torch.Tensor) -> torch.Tensor:
    r"""N̄(t) = θ(t)/π + 1: the smooth part of the zero count (Riemann–von Mangoldt)."""
    return theta(t) / math.pi + 1


def montgomery_r2(u: torch.Tensor) -> torch.Tensor:
    r"""Montgomery's pair correlation R₂(u) = 1 − (sin πu / πu)² (GUE)."""
    return 1 - torch.sinc(u) ** 2  # torch.sinc(u) = sin(πu)/(πu)


def wigner_gue(s: torch.Tensor) -> torch.Tensor:
    r"""Wigner surmise for GUE spacings p(s) = (32/π²) s² e^{−4s²/π} (within ~1% of the exact Gaudin law)."""
    return 32 / math.pi**2 * s**2 * torch.exp(-4 * s**2 / math.pi)


def zero_waves(x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    r"""2 Re(x^ρ/ρ) for every zero ρ_k = ½ + iγ_k and every x.   (X,), (K,) → (K, X)"""
    rho = 0.5 + 1j * gammas.to(torch.float64)
    return 2 * (torch.exp(rho[:, None] * torch.log(x)[None, :]) / rho[:, None]).real


def psi_smooth(x: torch.Tensor) -> torch.Tensor:
    r"""The zero-free part of the explicit formula: x − ln 2π − ½ ln(1 − x^{−2})."""
    return x - math.log(2 * math.pi) - 0.5 * torch.log(1 - x**-2)
