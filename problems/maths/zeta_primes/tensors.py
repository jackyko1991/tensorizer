"""Build the concrete tensors: the prime sieve, the Möbius function, ζ at the integers and the sample grids."""

import math

import torch

from .equation import theta, zeta

# Well-established values (Odlyzko's tables) used only to check the computation.
KNOWN_ZEROS = [14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588]
KNOWN_PI = {10: 4, 100: 25, 1000: 168, 10**4: 1229, 10**5: 9592, 10**6: 78498, 10**7: 664579}


def sieve(x_max: int, device: torch.device) -> torch.Tensor:
    """Sieve of Eratosthenes as a boolean tensor is_prime[n] for n = 0 … x_max: (x_max + 1,)."""
    is_prime = torch.ones(x_max + 1, dtype=torch.bool, device=device)
    is_prime[:2] = False
    for p in range(2, math.isqrt(x_max) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False  # strike every multiple from p² on in one slice
    return is_prime


def prime_pi(is_prime: torch.Tensor) -> torch.Tensor:
    """π(n) for n = 0 … x_max: the running count of primes.   (x_max + 1,) int64"""
    return torch.cumsum(is_prime.long(), dim=0)


def mobius(n_max: int) -> list[int]:
    """μ(n) for n = 0 … n_max (μ(0) unused): 0 if n has a square factor, else (−1)^(number of prime factors)."""
    mu = [1] * (n_max + 1)
    for p in range(2, n_max + 1):
        if all(p % q for q in range(2, math.isqrt(p) + 1)):  # p is prime
            for m in range(p, n_max + 1, p):
                mu[m] = -mu[m]
            for m in range(p * p, n_max + 1, p * p):
                mu[m] = 0
    return mu


def zeta_at_integers(k_max: int, device: torch.device) -> torch.Tensor:
    """ζ(k + 1) for k = 0 … k_max (index 0 unused), for Gram's series of R(x)."""
    s = torch.arange(1, k_max + 2, dtype=torch.float64, device=device).to(torch.complex128)
    s[0] = 2.0  # placeholder for the unused slot (ζ has a pole at s = 1)
    return zeta(s, 30).real


def height_for(n_zeros: int) -> float:
    """Height T with about n_zeros zeros below it, from the zero count N(T) ≈ θ(T)/π + 1 (bisection)."""
    lo, hi = 10.0, 10000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if theta(torch.tensor(mid, dtype=torch.float64)) / math.pi + 1 < n_zeros + 0.5 else (lo, mid)
    return hi + 1.0


def t_grid(t_max: float, step: float, device: torch.device) -> torch.Tensor:
    """Heights t on the critical line from 10 (below the first zero) to t_max: (T,)."""
    return torch.arange(10.0, t_max, step, dtype=torch.float64, device=device)
