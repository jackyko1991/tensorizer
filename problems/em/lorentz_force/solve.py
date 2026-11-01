"""Integrate the batched Lorentz-force ODE with 4th-order Runge–Kutta."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def solve(y0, q, m, E, B, eps, t_max: float, dt: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, P, 2, 3)).

    All P particles advance together: one RK4 step updates the whole (P, 2, 3) tensor.
    """
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda y, t: rhs(y, q, m, E, B, eps), y0, dt, n_steps)  # (T, P, 2, 3)
    times = torch.arange(n_steps + 1, device=y0.device) * dt
    return times, traj
