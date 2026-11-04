"""Build the concrete tensors: the prime sieve, Ulam and Sacks spiral coordinates and the family of quadratics."""

import math

import torch

# Well-established values used only to check the computation (OEIS A007508: twin pairs (p, p+2) with p ≤ 10^k).
KNOWN_TWINS = {10**3: 35, 10**4: 205, 10**5: 1224, 10**6: 8169, 10**7: 58980, 10**8: 440312}


def sieve(n_max: int, device: torch.device) -> torch.Tensor:
    """Sieve of Eratosthenes as a boolean tensor is_prime[n] for n = 0 … n_max: (n_max + 1,)."""
    is_prime = torch.ones(n_max + 1, dtype=torch.bool, device=device)
    is_prime[:2] = False
    for p in range(2, math.isqrt(n_max) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False  # strike every multiple from p² on in one strided slice
    return is_prime


def ulam_xy(n: torch.Tensor, start: int) -> torch.Tensor:
    """Grid position of every number n on the Ulam spiral that starts with `start` at (0, 0), all at once.

    With m = n − start + 1 the ring index is k = ⌈(√m − 1)/2⌉; ring k ends at (2k+1)² in the corner (k, −k) and is
    walked anticlockwise from (k, −k+1): up the right side, left along the top, down the left, right along the bottom.
    (N,) int → (N, 2) int64
    """
    m = (n - start + 1).long()
    k = torch.ceil((torch.sqrt(m.double()) - 1) / 2).long()
    side = 2 * k  # cells per side of ring k
    corner = (2 * k + 1) ** 2  # last number of ring k, at (k, −k)
    back = corner - m  # how far behind the corner, 0 … 8k − 1
    x = torch.where(back < side, k - back, torch.zeros_like(k))  # bottom side
    y = torch.where(back < side, -k, torch.zeros_like(k))
    on_left = (back >= side) & (back < 2 * side)
    x, y = torch.where(on_left, -k, x), torch.where(on_left, -k + back - side, y)
    on_top = (back >= 2 * side) & (back < 3 * side)
    x, y = torch.where(on_top, -k + back - 2 * side, x), torch.where(on_top, k, y)
    on_right = back >= 3 * side
    x, y = torch.where(on_right, k, x), torch.where(on_right, k - back + 3 * side, y)
    return torch.stack([x, y], dim=1)


def sacks_xy(n: torch.Tensor) -> torch.Tensor:
    """Sacks spiral: n at radius √n and angle 2π√n, so the squares k² line up on the positive x-axis.   (N,) → (N, 2)"""
    r = torch.sqrt(n.double())
    return torch.stack([r * torch.cos(2 * math.pi * r), r * torch.sin(2 * math.pi * r)], dim=1)


def quadratic_family(b_max: int, c_max: int, device: torch.device) -> torch.Tensor:
    """Coefficients (a, b, c) of f(k) = 4k² + bk + c, |b| ≤ b_max, 1 ≤ c ≤ c_max: every diagonal ray of the Ulam
    spiral is such a quadratic. Reducible ones (b² − 16c a perfect square) are dropped.   → (F, 3) int64"""
    b, c = torch.meshgrid(
        torch.arange(-b_max, b_max + 1, device=device), torch.arange(1, c_max + 1, device=device), indexing="ij"
    )
    b, c = b.flatten(), c.flatten()
    disc = b * b - 16 * c
    root = torch.sqrt(disc.clamp(min=0).double()).round().long()
    keep = ~((disc >= 0) & (root * root == disc))
    return torch.stack([torch.full_like(b, 4), b, c], dim=1)[keep]
