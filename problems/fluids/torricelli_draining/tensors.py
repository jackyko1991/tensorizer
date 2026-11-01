"""Build the concrete tensors: hole diameters and areas, and the initial water levels."""

import math

import torch


def hole_diameters(n: int, d_min: float, d_max: float, device: torch.device) -> torch.Tensor:
    """n hole diameters (m) spread evenly between d_min and d_max: shape (B,)."""
    if n == 1:
        return torch.tensor([d_min], device=device)
    return torch.linspace(d_min, d_max, n, device=device)


def circle_area(d: torch.Tensor | float) -> torch.Tensor | float:
    """Area π d²/4 of a circle of diameter d (works for tensors and plain numbers)."""
    return math.pi * d**2 / 4


def initial_levels(n: int, h0: float, device: torch.device) -> torch.Tensor:
    """Every tank starts full to height h0: shape (B,)."""
    return torch.full((n,), h0, device=device)
