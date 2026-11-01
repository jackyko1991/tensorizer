"""Find the zeros of Z(t) and rebuild π(x) from them with the explicit formula."""

import math

import torch

from problems._common.progress import steps

from .equation import hardy_z, li, tail_integral, zero_terms


def n_terms_for(t_max: float) -> int:
    """Euler–Maclaurin needs N well above |s|/2π; N = t/2 + 20 keeps the error below 1e-12."""
    return int(t_max / 2) + 20


def z_on_grid(t: torch.Tensor, n_terms: int, chunk: int = 4096) -> torch.Tensor:
    """Z(t) for a long grid, in chunks so the (T, N) term matrix stays small."""
    return torch.cat([hardy_z(t[i : i + chunk], n_terms) for i in range(0, len(t), chunk)])


def find_zeros(t: torch.Tensor, z: torch.Tensor, n_terms: int, n_bisect: int = 48) -> torch.Tensor:
    """Sign changes of Z on the grid bracket the zeros; all brackets are bisected at once.   → (K,) heights γ_k"""
    change = (z[:-1] * z[1:]) < 0
    lo, hi = t[:-1][change].clone(), t[1:][change].clone()
    z_lo = z[:-1][change].clone()
    for _ in steps(n_bisect, "bisection"):
        mid = 0.5 * (lo + hi)
        z_mid = hardy_z(mid, n_terms)
        same = torch.sign(z_mid) == torch.sign(z_lo)
        lo, z_lo = torch.where(same, mid, lo), torch.where(same, z_mid, z_lo)
        hi = torch.where(same, hi, mid)
    return 0.5 * (lo + hi)


def prime_count_from_zeros(x: torch.Tensor, gammas: torch.Tensor, mu: list[int]) -> torch.Tensor:
    """π_K(x) = Σ_n μ(n)/n · J_K(x^{1/n}) using the first K = 0 … len(gammas) zeros, all at once.

    J_K(y) = li(y) − Σ_{k≤K} 2 Re li(y^{ρ_k}) − ln 2 + ∫_y^∞ dt/(t(t²−1) ln t), and J(y) = 0 for y < 2.
    Returns (K + 1, X): row K is the approximation with K zeros; row 0 has none (the smooth part, close to R(x)).
    """
    out = torch.zeros(len(gammas) + 1, len(x), dtype=torch.float64, device=x.device)
    for n in range(1, int(math.log2(x.max().item())) + 1):
        if mu[n] == 0:
            continue
        y = x ** (1 / n)
        smooth = li(y) - math.log(2) + tail_integral(y)  # (X,)
        waves = torch.cumsum(zero_terms(y, gammas), dim=0)  # (K, X): sum over the first k zeros
        j = torch.cat([smooth[None], smooth[None] - waves])  # (K + 1, X)
        out += mu[n] / n * torch.where(y >= 2, j, 0.0)
    return out
