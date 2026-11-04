"""Find the zeros of Z(t), count them, unfold them and feed them into the explicit formula."""

import torch

from problems._common.progress import steps

from .equation import psi_smooth, riemann_siegel_z, smooth_count, zero_waves


def z_on_grid(t: torch.Tensor, chunk: int = 20000) -> torch.Tensor:
    """Z(t) on a long grid, in chunks so the (T, N) term matrix stays small."""
    return torch.cat([riemann_siegel_z(t[i : i + chunk]) for i in range(0, len(t), chunk)])


def find_zeros(t: torch.Tensor, z: torch.Tensor, n_bisect: int = 32) -> torch.Tensor:
    """Sign changes of Z bracket the zeros on the critical line; all brackets are bisected at once.   → (K,)"""
    change = (z[:-1] * z[1:]) < 0
    lo, hi, z_lo = t[:-1][change], t[1:][change], z[:-1][change]
    for _ in steps(n_bisect, "bisection"):
        mid = 0.5 * (lo + hi)
        z_mid = riemann_siegel_z(mid)
        same = torch.sign(z_mid) == torch.sign(z_lo)
        lo, z_lo, hi = torch.where(same, mid, lo), torch.where(same, z_mid, z_lo), torch.where(same, hi, mid)
    return 0.5 * (lo + hi)


def count_below(gammas: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """N(t) = number of zeros with 0 < γ ≤ t (the first zero is at 14.13, so the grid from 10 misses none)."""
    return torch.searchsorted(gammas, t.contiguous(), right=True)


def s_function(gammas: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """S(t) = N(t) − θ(t)/π − 1: the fluctuating part of the zero count.   (T,)"""
    return count_below(gammas, t) - smooth_count(t)


def gram_violations(z_at_gram: torch.Tensor) -> torch.Tensor:
    """Indices n where Gram's law (−1)^n Z(g_n) > 0 fails ("bad" Gram points)."""
    n = torch.arange(len(z_at_gram), device=z_at_gram.device)
    sign = 1 - 2 * (n % 2)
    return torch.nonzero(sign * z_at_gram <= 0).flatten()


def unfold(gammas: torch.Tensor) -> torch.Tensor:
    """x_k = θ(γ_k)/π + 1: zeros rescaled to mean spacing 1 (the local density is ln(t/2π)/2π)."""
    return smooth_count(gammas)


def pair_differences(x: torch.Tensor, neighbours: int = 10) -> torch.Tensor:
    """x_{k+j} − x_k for j = 1 … J and every k: the pair correlation needs only nearby pairs.   (K − J, J)"""
    k = len(x) - neighbours
    j = torch.arange(1, neighbours + 1, device=x.device)
    idx = torch.arange(k, device=x.device)[:, None] + j[None, :]  # (K − J, J)
    return x[idx] - x[:k, None]


def pair_correlation(x: torch.Tensor, edges: torch.Tensor, neighbours: int = 10) -> torch.Tensor:
    """Density of differences u = x_j − x_k (j > k) per zero and per unit u, on the bins `edges`: R₂(u)."""
    d = pair_differences(x, neighbours).flatten()
    counts = torch.histc(d, bins=len(edges) - 1, min=edges[0].item(), max=edges[-1].item())
    return counts / ((len(x) - neighbours) * (edges[1] - edges[0]))


def psi_from_zeros(x: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    """ψ_K(x) = x − Σ_{k ≤ K} 2 Re(x^{ρ_k}/ρ_k) − ln 2π − ½ ln(1 − x^{−2}) for every K = 0 … len(gammas).
    Returns (K + 1, X); row K uses the first K zeros."""
    smooth = psi_smooth(x)  # (X,)
    waves = torch.cumsum(zero_waves(x, gammas), dim=0)  # (K, X)
    return torch.cat([smooth[None], smooth[None] - waves])
