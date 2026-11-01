"""Build the concrete tensors: random darts in the unit square and random needles on a ruled floor."""

import math

import torch


def darts(n_experiments: int, n_darts: int, seed: int, device: torch.device) -> torch.Tensor:
    """x_{bni}: dart n of experiment b, uniform in the unit square [0, 1)²: shape (B, N, 2)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n_experiments, n_darts, 2, generator=gen, dtype=torch.float64).to(device)


def needles(n_experiments: int, n_needles: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Needle centres' distance y to the nearest line, uniform in [0, d/2) with d = 1, and angles φ in [0, π/2):
    two tensors of shape (B, N)."""
    gen = torch.Generator().manual_seed(seed + 1)
    y = 0.5 * torch.rand(n_experiments, n_needles, generator=gen, dtype=torch.float64)
    phi = 0.5 * math.pi * torch.rand(n_experiments, n_needles, generator=gen, dtype=torch.float64)
    return y.to(device), phi.to(device)
