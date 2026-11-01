"""Step the whole cloth forward with 4th-order Runge–Kutta, starting from rest."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def solve(
    x0: torch.Tensor, D: torch.Tensor, rest: torch.Tensor, free: torch.Tensor, cloth: dict, t_max: float, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, 2, N, 3)) with positions in [:, 0] and velocities in [:, 1].

    The springs make the system stiff: dt must stay well below √(m/k), the period of one mass on one spring.
    """
    n_steps = int(round(t_max / dt))
    y0 = torch.stack([x0, torch.zeros_like(x0)])  # (2, N, 3) released from rest
    traj = integrate(lambda y, t: rhs(y, D, rest, free, cloth), y0, dt, n_steps)
    times = torch.arange(n_steps + 1, device=x0.device, dtype=x0.dtype) * dt
    return times, traj
