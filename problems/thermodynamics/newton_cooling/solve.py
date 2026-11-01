"""Integrate the batch of cooling ODEs with 4th-order Runge–Kutta."""

import torch

from problems._common.integrators import integrate

from .equation import cooling_rate


def solve(
    T0: torch.Tensor, k: torch.Tensor, T_env: float, t_max: float, dt: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,) in minutes, temperatures (T, B) in °C).

    One RK4 step updates the whole (B,) tensor: every cup cools at the same time.
    """
    n_steps = int(round(t_max / dt))
    temps = integrate(lambda T, t: cooling_rate(T, k, T_env), T0, dt, n_steps)  # (T, B)
    times = torch.arange(n_steps + 1, device=T0.device) * dt
    return times, temps
