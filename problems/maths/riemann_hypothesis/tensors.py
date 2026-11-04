"""Build the concrete tensors: the height grid, Gram points and the prime-power staircase ψ(x)."""

import math

import torch

from .equation import theta

# Published zeros (Odlyzko's tables, reproduced with mpmath.zetazero) used only to check the computation.
KNOWN_ZEROS = {1: 14.134725142, 2: 21.022039639, 3: 25.010857580, 100: 236.524229666, 1000: 1419.422480946}


def t_grid(t_max: float, step: float, device: torch.device) -> torch.Tensor:
    """Heights on the critical line from 10 (below the first zero) to t_max: (T,)."""
    return torch.arange(10.0, t_max, step, dtype=torch.float64, device=device)


def gram_points(n_max: int, device: torch.device) -> torch.Tensor:
    """Gram points g_n, θ(g_n) = nπ for n = 0 … n_max, by Newton's method on all of them at once: (n_max + 1,).
    θ′(t) ≈ ½ ln(t/2π), and θ is increasing and convex for t > 7, so Newton converges from any start there."""
    n = torch.arange(n_max + 1, dtype=torch.float64, device=device)
    g = 20 + 2 * math.pi * n / torch.log(n + 3)  # rough start
    for _ in range(40):
        g = g - (theta(g) - n * math.pi) / (0.5 * torch.log(g / (2 * math.pi)))
    return g


def chebyshev_psi(x_max: int, device: torch.device) -> torch.Tensor:
    """ψ(n) = Σ_{p^k ≤ n} ln p for n = 0 … x_max: von Mangoldt's Λ placed on the prime powers, then a cumulative
    sum.   (x_max + 1,)"""
    is_prime = torch.ones(x_max + 1, dtype=torch.bool, device=device)
    is_prime[:2] = False
    for p in range(2, math.isqrt(x_max) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    primes = torch.nonzero(is_prime).flatten()  # (P,)
    k = torch.arange(1, int(math.log2(max(x_max, 2))) + 1, device=device)  # exponents 1 … log₂ x
    powers = primes[:, None].double() ** k[None, :]  # (P, K) every prime power p^k
    keep = powers <= x_max
    lam = torch.zeros(x_max + 1, dtype=torch.float64, device=device)
    lam[powers[keep].long()] = torch.log(primes[:, None].double().expand_as(powers))[keep]  # Λ(p^k) = ln p
    return torch.cumsum(lam, dim=0)
