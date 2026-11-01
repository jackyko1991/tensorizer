"""Integrate both double pendulums together with 4th-order Runge–Kutta."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def solve(
    y0: torch.Tensor, m1: float, m2: float, L1: float, L2: float, g: float, t_max: float, dt: float = 0.004
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, B, 2, 2)).

    Every RK4 stage builds the (B, 2, 2) mass matrix and calls torch.linalg.solve once for the whole batch.
    """
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda y, t: rhs(y, m1, m2, L1, L2, g), y0, dt, n_steps)
    times = torch.arange(n_steps + 1, device=y0.device, dtype=y0.dtype) * dt
    return times, traj


def positions(q: torch.Tensor, L1: float, L2: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Cartesian positions of both bobs from the angles q (..., 2) → two tensors (..., 2) holding (x, y)."""
    bob1 = L1 * torch.stack([torch.sin(q[..., 0]), -torch.cos(q[..., 0])], dim=-1)
    bob2 = bob1 + L2 * torch.stack([torch.sin(q[..., 1]), -torch.cos(q[..., 1])], dim=-1)
    return bob1, bob2
