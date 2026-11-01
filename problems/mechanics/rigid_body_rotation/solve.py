"""Integrate Euler's equations together with the orientation, using 4th-order Runge–Kutta."""

import torch

from problems._common.integrators import integrate

from .equation import levi_civita, rhs


def solve(y0: torch.Tensor, inertia: torch.Tensor, t_max: float, dt: float = 0.004) -> dict[str, torch.Tensor]:
    """Return times (T,), orientations R (T, 3, 3) and body-frame angular velocities ω (T, 3)."""
    eps = levi_civita(y0.dtype, y0.device)  # (3, 3, 3)
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda y, t: rhs(y, inertia, eps), y0, dt, n_steps)  # (T, 4, 3)
    times = torch.arange(n_steps + 1, device=y0.device, dtype=y0.dtype) * dt
    return {"times": times, "R": traj[:, :3], "omega": traj[:, 3]}
