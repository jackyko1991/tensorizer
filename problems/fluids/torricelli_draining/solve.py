"""Integrate all tanks together with RK4 until the slowest one is empty."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def solve(
    h0: torch.Tensor, a: torch.Tensor, A: float, g: float, t_max: float, n_steps: int = 2000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), levels (T, B)). Levels are clamped at 0 once a tank has emptied."""
    dt = t_max / n_steps
    levels = integrate(lambda h, t: rhs(h, a, A, g), h0, dt, n_steps).clamp(min=0.0)
    times = torch.arange(n_steps + 1, device=h0.device) * dt
    return times, levels


def numerical_drain_time(times: torch.Tensor, levels: torch.Tensor, h_small: float = 1e-6) -> torch.Tensor:
    """First time each level drops below h_small: (B,)."""
    empty = levels < h_small  # (T, B)
    return times[empty.int().argmax(dim=0)]
