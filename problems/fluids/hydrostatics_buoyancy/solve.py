"""Integrate all blocks together with RK4; a block that sinks comes to rest on the tank floor."""

import torch

from problems._common.integrators import rk4_step
from problems._common.progress import steps

from .equation import rhs


def solve(
    y0: torch.Tensor,
    rho_b: torch.Tensor,
    s: float,
    depth: float,
    t_max: float,
    g: float = 9.81,
    rho_w: float = 1000.0,
    gamma: float = 4.0,
    dt: float = 0.002,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, B, 2)).

    After every step the floor pushes back: a block below z = -depth is put on the floor with its
    downward velocity removed (the normal force does this in reality).
    """
    n_steps = int(round(t_max / dt))
    y = y0.clone()
    ys = [y]
    for n in steps(n_steps, "RK4 steps"):
        y = rk4_step(lambda y, t: rhs(y, rho_b, s, g, rho_w, gamma), y, n * dt, dt)
        on_floor = y[:, 0] < -depth  # (B,) bool
        y[on_floor, 0] = -depth  # sit on the floor
        y[on_floor, 1] = y[on_floor, 1].clamp(min=0.0)  # no further sinking
        ys.append(y)
    times = torch.arange(n_steps + 1, device=y0.device) * dt
    return times, torch.stack(ys)


def floating_fraction(traj: torch.Tensor, s: float) -> torch.Tensor:
    """Fraction of each block under water at the end of the run: (B,). Archimedes predicts ρ_b / ρ_w."""
    return torch.clamp(-traj[-1, :, 0], 0.0, s) / s
