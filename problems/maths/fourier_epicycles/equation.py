r"""Epicycles → tensor translation: the complex discrete Fourier transform of a closed curve.

Samples z_n (n = 0 … N−1) at times t_n = n/N; frequencies k = −N/2 … N/2 − 1 (integers, the turns per trip).

    W_{kn} = e^{-2\pi i k n / N},        c_k = \tfrac1N W_{kn} z_n                       (n summed)    the DFT
    z(t) = c_k e^{2\pi i k t}                                                           (k summed)    the inverse

Each term c_k e^{2πikt} is a circle of radius |c_k| turning k times per trip, starting at angle arg c_k; adding them
tip to tail is a chain of epicycles. Keep only the M largest circles (rank r_k < M, sorted by radius):

    S_{mt} = [r_k < M_m]\, c_k\, e^{2\pi i k t}                                         (k summed)
    E_m = \tfrac1N |z_n - S_{mn}|^2 = \sum_{r_k \ge M_m} |c_k|^2                          (Parseval)
"""

import math

import torch


def frequencies(n: int, device: torch.device) -> torch.Tensor:
    """k = −N/2 … N/2 − 1 (N,) int64: negative k turn clockwise, positive anticlockwise."""
    return torch.arange(n, device=device) - n // 2


def dft(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """c_k = (1/N) W_{kn} z_n with W_{kn} = exp(−2πi k n / N).  z (N,) complex → c (N,) complex, k (N,). n summed."""
    n = z.shape[0]
    k = frequencies(n, z.device)
    idx = torch.arange(n, device=z.device)
    W = torch.exp(-2j * math.pi * torch.outer(k, idx).to(torch.float64) / n)  # (N, N) the DFT matrix
    return torch.einsum("kn,n->k", W, z) / n, k


def idft(c: torch.Tensor, k: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """z(t) = c_k exp(2πi k t) at any times t (T,) → (T,) complex.  k summed."""
    E = torch.exp(2j * math.pi * torch.outer(t, k.to(torch.float64)))  # (T, K) one rotating unit vector per term
    return torch.einsum("tk,k->t", E, c)


def by_radius(c: torch.Tensor) -> torch.Tensor:
    """Indices of the terms from the largest circle |c_k| to the smallest (N,) int64."""
    return torch.argsort(c.abs(), descending=True, stable=True)


def partial_sums(c: torch.Tensor, k: torch.Tensor, order: torch.Tensor, m: torch.Tensor, t: torch.Tensor):
    """S_{mt}: the curve drawn by only the M_m largest circles, for every M in m (M,) at once → (M, T) complex.

    keep_{mk} = [rank of term k < M_m] is a 0/1 matrix; S = keep · (c_k e^{2πikt}), k summed.
    """
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.shape[0], device=order.device)  # (K,) rank of each term by radius
    keep = (rank[None, :] < m[:, None]).to(c.dtype)  # (M, K)
    E = torch.exp(2j * math.pi * torch.outer(t, k.to(torch.float64)))  # (T, K)
    return torch.einsum("mk,k,tk->mt", keep, c, E)


def chain(c: torch.Tensor, k: torch.Tensor, order: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Centres of the epicycles at times t: (T, K + 1) complex, from 0 (the fixed centre) to the pen tip.

    Term j of the chain is c_{o_j} e^{2πi k_{o_j} t}, biggest first; the running sum gives each circle's centre.
    """
    terms = c[order] * torch.exp(2j * math.pi * torch.outer(t, k[order].to(torch.float64)))  # (T, K)
    return torch.cat([terms.new_zeros(t.shape[0], 1), torch.cumsum(terms, dim=1)], dim=1)


def mean_power(z: torch.Tensor) -> torch.Tensor:
    """(1/N) Σ_n |z_n|², which Parseval says equals Σ_k |c_k|²."""
    return (z.abs() ** 2).mean()


def sample_error(z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """E_m = (1/N) Σ_n |z_n − S_{mn}|²: mean squared distance at the sample points, (M, N) → (M,)."""
    return ((s - z).abs() ** 2).mean(-1)
