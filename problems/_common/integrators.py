"""Time integrators that work on tensors of any shape (batches included)."""

from __future__ import annotations

from typing import Callable

import torch

from .progress import steps

Rhs = Callable[[torch.Tensor, float], torch.Tensor]


def rk4_step(f: Rhs, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
    """One classical Runge–Kutta step for dy/dt = f(y, t)."""
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(f: Rhs, y0: torch.Tensor, dt: float, n_steps: int) -> torch.Tensor:
    """Return the whole trajectory, shape (n_steps + 1, *y0.shape)."""
    ys = [y0]
    y, t = y0, 0.0
    for _ in steps(n_steps, "RK4 steps"):
        y = rk4_step(f, y, t, dt)
        t += dt
        ys.append(y)
    return torch.stack(ys)
