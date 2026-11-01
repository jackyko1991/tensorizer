"""Integrate the evaporation dM/dt = −α/M² with RK4, letting each step remove a fixed fraction of the mass."""

import torch

from problems._common.progress import steps

from .equation import mass_loss_rate


def rk4_step(M: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """One RK4 step of dM/dt = f(M) (f does not depend on t), elementwise for a batch of holes."""
    k1 = mass_loss_rate(M)
    k2 = mass_loss_rate(M + 0.5 * dt * k1)
    k3 = mass_loss_rate(M + 0.5 * dt * k2)
    k4 = mass_loss_rate(M + dt * k3)
    return M + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def evaporate(M0: torch.Tensor, fraction: float = 0.01, M_end: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T, B), masses (T, B)) from M0 down to M_end·M0.

    The time step shrinks with the hole: dt = fraction · M / |dM/dt|, so every step burns about the same
    fraction of the remaining mass. A fixed dt would either crawl at the start or overshoot in the final burst.
    """
    n_steps = int(torch.log(torch.tensor(1 / M_end)) / fraction) + 1
    t, M = torch.zeros_like(M0), M0.clone()
    times, masses = [t], [M]
    for _ in steps(n_steps, "RK4 steps"):
        dt = fraction * M / mass_loss_rate(M).abs()
        M = rk4_step(M, dt)
        t = t + dt
        times.append(t)
        masses.append(M)
    return torch.stack(times), torch.stack(masses)
