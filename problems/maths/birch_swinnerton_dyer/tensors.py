"""Build the concrete tensors: five reference elliptic curves of rank 0 … 4 and the primes."""

import math

import torch

# Smallest-conductor curves of each rank (Cremona / LMFDB labels), y² + a1·xy + a3·y = x³ + a2·x² + a4·x + a6.
CURVES = {
    "11a1": (0, -1, 1, -10, -20),
    "37a1": (0, 0, 1, -1, 0),
    "389a1": (0, 1, 1, -2, 0),
    "5077a1": (0, 0, 1, -7, 6),
    "234446a1": (1, -1, 0, -79, 289),
}
RANKS = [0, 1, 2, 3, 4]  # Mordell–Weil ranks over Q (known, e.g. from explicit generators and descent)
CONDUCTORS = [11, 37, 389, 5077, 234446]


def curve_coefficients(device: torch.device) -> torch.Tensor:
    """[a1, a2, a3, a4, a6] for every curve: (C, 5) int64."""
    return torch.tensor(list(CURVES.values()), dtype=torch.int64, device=device)


def primes_up_to(x_max: int, device: torch.device) -> torch.Tensor:
    """All primes p ≤ x_max (sieve of Eratosthenes): (P,) int64."""
    is_prime = torch.ones(x_max + 1, dtype=torch.bool, device=device)
    is_prime[:2] = False
    for p in range(2, math.isqrt(x_max) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    return torch.nonzero(is_prime).flatten()
