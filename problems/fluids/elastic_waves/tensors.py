"""Build the concrete tensors: the displacement field, the point-force source and the absorbing border."""

import math

import torch


def at_rest(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Displacement u_i at the current and previous time step, both zero: (2, N, N) each."""
    return torch.zeros(2, n, n, device=device), torch.zeros(2, n, n, device=device)


def point_force(n: int, direction_deg: float, width: float, device: torch.device) -> torch.Tensor:
    """Spatial pattern of the source: a small Gaussian blob at the centre pushing along one direction.

    Returns f_i(x, y) of shape (2, N, N) with unit peak strength.
    """
    c = (n - 1) / 2
    x = torch.arange(n, device=device, dtype=torch.float32) - c
    blob = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * width**2))  # (N, N)
    a = math.radians(direction_deg)
    return torch.stack([math.cos(a) * blob, math.sin(a) * blob])


def ricker(t: float, f0: float) -> float:
    """Ricker (Mexican-hat) wavelet in time with peak frequency f0, centred at t0 = 1.2 / f0."""
    s = (math.pi * f0 * (t - 1.2 / f0)) ** 2
    return (1 - 2 * s) * math.exp(-s)


def sponge(n: int, width: int, strength: float, device: torch.device) -> torch.Tensor:
    """Damping factor (N, N): 1 in the interior, smoothly below 1 near the edges so outgoing waves fade away."""
    i = torch.arange(n, device=device, dtype=torch.float32)
    depth = torch.clamp(width - torch.minimum(i, n - 1 - i), min=0) / width  # 0 inside, → 1 at the edge
    d1 = 1 - strength * depth**2
    return d1[:, None] * d1[None, :]
