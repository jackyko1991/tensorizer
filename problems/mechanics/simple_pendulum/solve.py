"""Integrate all pendulums together with 4th-order Runge–Kutta and measure their periods."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def solve(y0: torch.Tensor, g: float, L: float, t_max: float, dt: float = 0.005) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, B, 2)). One RK4 step updates all B pendulums."""
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda y, t: rhs(y, g, L), y0, dt, n_steps)  # (T, B, 2)
    times = torch.arange(n_steps + 1, device=y0.device) * dt
    return times, traj


def periods(times: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Period of each pendulum from its first downward zero crossing: (T, B) → (B,).

    Released from rest at θ > 0, a pendulum reaches θ = 0 after exactly a quarter period.
    We find the step k where θ changes sign and interpolate linearly between k and k + 1.
    """
    crossed = (theta[:-1] > 0) & (theta[1:] <= 0)  # (T-1, B)
    k = crossed.int().argmax(dim=0)  # (B,) first crossing step
    b = torch.arange(theta.shape[1])
    before, after = theta[k, b], theta[k + 1, b]
    dt = times[1] - times[0]
    quarter = times[k] + dt * before / (before - after)
    return 4 * quarter
