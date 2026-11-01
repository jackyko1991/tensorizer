"""Evaluate the steady flow on the axis and move the tracers through it with RK4."""

import torch

from problems._common.integrators import integrate

from . import tensors
from .equation import area, pressure, velocity


def steady_flow(x: torch.Tensor, r1: float, r2: float, u1: float, p1: float, rho: float) -> dict:
    """Radius, area, velocity and pressure along the axis, all (N,), plus the flow rate Q."""
    r = tensors.radius(x, r1, r2)
    A = area(r)
    Q = u1 * area(torch.tensor(r1)).item()  # flow rate fixed by the inlet
    u = velocity(Q, A)
    return {"r": r, "A": A, "u": u, "p": pressure(u, p1, u1, rho), "Q": Q}


def move_tracers(
    x0: torch.Tensor, r1: float, r2: float, Q: float, t_max: float, dt: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), positions (T, n)) for dx_n/dt = u(x_n) = Q / A(x_n), all tracers at once."""
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda x, t: velocity(Q, area(tensors.radius(x, r1, r2))), x0, dt, n_steps)
    times = torch.arange(n_steps + 1, device=x0.device) * dt
    return times, traj % tensors.LENGTH  # a tracer leaving at the outlet re-enters at the inlet
