"""Build the concrete tensors: synthetic point clouds (ring, two blobs, spiral), weights and Gaussian samples.

Every shape is generated here from a formula and random numbers, no images are used.
"""

import math

import torch


def ring(n: int, seed: int, device: torch.device) -> torch.Tensor:
    """n points on a noisy circle of radius 1: (n, 2)."""
    gen = torch.Generator().manual_seed(seed)
    t = 2 * math.pi * torch.rand(n, generator=gen, dtype=torch.float64)
    r = 1 + 0.06 * torch.randn(n, generator=gen, dtype=torch.float64)
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], -1).to(device)


def two_blobs(n: int, seed: int, device: torch.device) -> torch.Tensor:
    """n points in two Gaussian blobs at (±0.9, 0.2), the left one wider: (n, 2)."""
    gen = torch.Generator().manual_seed(seed)
    side = (torch.arange(n) % 2).double() * 2 - 1  # −1, +1, −1, …
    z = torch.randn(n, 2, generator=gen, dtype=torch.float64)
    scale = torch.where(side < 0, 0.32, 0.22)[:, None]
    return (z * scale + torch.stack([0.9 * side, torch.full_like(side, 0.2)], -1)).to(device)


def spiral(n: int, seed: int, device: torch.device) -> torch.Tensor:
    """n points along an Archimedean spiral r = 0.12 + 0.95 s, angle 3π s, s ∈ [0, 1]: (n, 2)."""
    gen = torch.Generator().manual_seed(seed)
    s = torch.rand(n, generator=gen, dtype=torch.float64).sqrt()  # more points on the long outer arms
    r, t = 0.12 + 0.95 * s, 3 * math.pi * s
    pts = torch.stack([r * torch.cos(t), r * torch.sin(t)], -1)
    return (pts + 0.03 * torch.randn(n, 2, generator=gen, dtype=torch.float64)).to(device)


SHAPES = {"ring": ring, "two blobs": two_blobs, "spiral": spiral}


def uniform_weights(n: int, device: torch.device) -> torch.Tensor:
    """a_i = 1/n: (n,) float64."""
    return torch.full((n,), 1.0 / n, dtype=torch.float64, device=device)


def epsilons(eps_min: float, eps_max: float, count: int, device: torch.device) -> torch.Tensor:
    """count values of ε, log-spaced from eps_max down to eps_min: (E,)."""
    return torch.logspace(math.log10(eps_max), math.log10(eps_min), count, dtype=torch.float64, device=device)


def gaussian_samples(m: torch.Tensor, s: torch.Tensor, n: int, device: torch.device, seed: int) -> torch.Tensor:
    """n samples of N(m, S) as m + L z with S = L Lᵀ (Cholesky): (n, d)."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n, len(m), generator=gen, dtype=torch.float64).to(device)
    return m + z @ torch.linalg.cholesky(s).T
