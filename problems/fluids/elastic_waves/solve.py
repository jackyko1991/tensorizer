"""Leapfrog time stepping of the Navier–Cauchy equation, saving div u and curl u."""

import torch

from problems._common.progress import steps

from . import tensors
from .equation import acceleration, curl, divergence


def simulate(
    u: torch.Tensor, u_prev: torch.Tensor, force: torch.Tensor, damping: torch.Tensor, h: float, dt: float,
    lam: float, mu: float, rho: float, f0: float, t_max: float,
) -> dict:  # fmt: skip
    """u^{n+1} = 2u^n - u^{n-1} + Δt² (\\ddot u + f/ρ), damped near the border.

    Returns times (T,), div u (T, N, N) and curl u (T, N, N).
    """
    n_steps = int(t_max / dt)
    times, divs, curls = [0.0], [divergence(u, h)], [curl(u, h)]
    for n in steps(n_steps, "time steps"):
        source = tensors.ricker(n * dt, f0) * force / rho
        u_next = 2 * u - u_prev + dt**2 * (acceleration(u, h, lam, mu, rho) + source)
        u_prev, u = u * damping, u_next * damping
        times.append((n + 1) * dt)
        divs.append(divergence(u, h))
        curls.append(curl(u, h))
    return {"times": torch.tensor(times), "div": torch.stack(divs), "curl": torch.stack(curls), "u": u}


def front_radius(field: torch.Tensor, direction: str) -> float:
    """Distance from the centre to the strongest |field| along the +x axis or the +y axis (in cells)."""
    n = field.shape[-1]
    c = n // 2
    line = field[c:, c] if direction == "x" else field[c, c:]
    return float(line.abs().argmax())
