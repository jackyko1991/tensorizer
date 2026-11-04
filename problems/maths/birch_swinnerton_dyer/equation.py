r"""Mathematics → tensor translation.

1. Point counting. For odd p, completing the square turns y² + a1xy + a3y = x³ + a2x² + a4x + a6 into
   Y² = f(x) = 4x³ + b2x² + 2b4x + b6 (Y = 2y + a1x + a3), so #E(F_p) = p + 1 + Σ_x χ_p(f(x)) with the Legendre symbol
   χ_p(v) = 1 (nonzero square), −1 (non-square), 0 (v ≡ 0). Hence the Frobenius trace

       a_p = p + 1 − #E(F_p) = −Σ_{x ∈ F_p} χ_p(f(x)),

   evaluated for a whole block of primes and every x at once: χ from a table of squares y² mod p (scatter).

2. The L-function L(E, s) = Σ a_n n^{−s} (a_n multiplicative, a_{p^k} = a_p a_{p^{k−1}} − p a_{p^{k−2}} for good p)
   satisfies Λ(s) = (√N/2π)^s Γ(s) L(E, s) = ε Λ(2 − s) (modularity: Wiles, Taylor–Wiles, BCDT). With
   F(t) = Σ_n a_n e^{−2πnt/√N}:

       Λ(s) = ∫_1^∞ F(t) (t^{s−1} + ε t^{1−s}) dt,    L^{(r)}(E, 1) = (2π/√N) · 2 ∫_1^∞ F(t) (ln t)^r dt  if ε = (−1)^r

   and the integrals become an einsum over a (n, u) grid with t = e^u.
"""

import math

import numpy as np
import torch


def b_invariants(coeffs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """b2, b4, b6, b8 of each curve.   (C, 5) → four (C,) tensors"""
    a1, a2, a3, a4, a6 = coeffs.unbind(-1)
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    return b2, b4, b6, b8


def discriminant(coeffs: torch.Tensor) -> torch.Tensor:
    """Δ = −b2²b8 − 8b4³ − 27b6² + 9b2b4b6; the curve is singular mod p exactly when p | Δ.   (C,)"""
    b2, b4, b6, b8 = b_invariants(coeffs)
    return -b2 * b2 * b8 - 8 * b4**3 - 27 * b6 * b6 + 9 * b2 * b4 * b6


def frobenius_traces(coeffs: torch.Tensor, primes: torch.Tensor, block: int = 3_000_000) -> torch.Tensor:
    r"""a_p = −Σ_x χ_p(f(x)) for every curve and prime (bad primes included: then a_p ∈ {0, ±1}).
    coeffs: (C, 5), primes: (P,) → (C, P) int64. Primes are processed in blocks of about `block` (p, x) pairs."""
    b2, b4, b6, _ = (b[:, None, None] for b in b_invariants(coeffs))  # (C, 1, 1)
    out, start = [], 0
    while start < len(primes):
        stop = start + 1
        while stop < len(primes) and (stop + 1 - start) * int(primes[stop]) <= block:
            stop += 1
        p = primes[start:stop]  # (Pb,)
        width = int(p.max())
        x = torch.arange(width, device=p.device)  # (X,)
        inside = x[None, :] < p[:, None]  # (Pb, X): only x ∈ F_p
        pm = p[:, None]
        is_square = torch.zeros(len(p), width, dtype=torch.bool, device=p.device)  # (Pb, X) squares mod p
        is_square.scatter_(1, torch.where(inside, x * x % pm, 0), True)
        xr = x[None, None, :] % pm[None]  # (1, Pb, X)
        f = (((4 * xr + b2) % pm * xr + 2 * b4) % pm * xr + b6) % pm  # Horner: 4x³ + b2x² + 2b4x + b6 mod p
        chi = torch.where(torch.gather(is_square.expand(len(coeffs), -1, -1), 2, f), 1, -1)  # (C, Pb, X)
        chi = torch.where((f == 0) | ~inside[None], 0, chi)
        a_p = -chi.sum(dim=-1)
        two = p == 2  # completing the square needs 2 to be invertible: count F_2 points directly
        if two.any():
            a_p[:, two] = (2 - _affine_count_mod2(coeffs))[:, None]
        out.append(a_p)
        start = stop
    return torch.cat(out, dim=1)


def _affine_count_mod2(coeffs: torch.Tensor) -> torch.Tensor:
    """Number of (x, y) ∈ F_2² on each curve, by checking all four pairs.   (C,)"""
    a1, a2, a3, a4, a6 = (c[:, None] for c in coeffs.unbind(-1))
    x = torch.tensor([0, 0, 1, 1], device=coeffs.device)
    y = torch.tensor([0, 1, 0, 1], device=coeffs.device)
    lhs = y * y + a1 * x * y + a3 * y - (x**3 + a2 * x * x + a4 * x + a6)
    return (lhs % 2 == 0).sum(dim=1)


def real_period(coeffs: torch.Tensor) -> torch.Tensor:
    r"""Ω = ∫_{E(R)} |dx / (2y + a1x + a3)| = c_∞ · 2∫_{e1}^∞ dx/√f(x), with e1 the largest real root of
    f = 4x³ + b2x² + 2b4x + b6 and c_∞ = 2 when E(R) has two components (Δ > 0). Substituting x = e1 + tan²φ
    removes the endpoint singularity.   (C, 5) → (C,)"""
    b2, b4, b6, _ = (b.double() for b in b_invariants(coeffs))
    companion = torch.zeros(len(coeffs), 3, 3, dtype=torch.float64, device=coeffs.device)
    companion[:, 0] = torch.stack([-b2 / 4, -2 * b4 / 4, -b6 / 4], dim=1)
    companion[:, 1, 0] = companion[:, 2, 1] = 1.0
    roots = torch.linalg.eigvals(companion)  # (C, 3) roots of f/4
    e1 = torch.where(roots.imag.abs() < 1e-9, roots.real, -math.inf).max(dim=1).values  # largest real root
    nodes, weights = (torch.tensor(v, device=coeffs.device) for v in np.polynomial.legendre.leggauss(200))
    phi = math.pi / 4 * (nodes + 1)  # Gauss–Legendre nodes on (0, π/2): (Φ,)
    w = torch.tan(phi)
    x = e1[:, None] + w * w  # (C, Φ)
    f = ((4 * x + b2[:, None]) * x + 2 * b4[:, None]) * x + b6[:, None]
    integrand = 2 * w / torch.cos(phi) ** 2 / torch.sqrt(f)  # dx = 2w sec²φ dφ
    half = math.pi / 4 * integrand @ weights  # ∫_{e1}^∞ dx/√f; the integrand is smooth on [0, π/2]
    components = torch.where(discriminant(coeffs) > 0, 2.0, 1.0)
    return components * 2 * half
