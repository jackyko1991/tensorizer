"""Integrate all planets at once with RK4, then read the period of each orbit off the trajectory."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def solve(y0: torch.Tensor, GM: float, t_max: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, B, 2, 2)). One RK4 step moves every planet."""
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda y, t: rhs(y, GM), y0, dt, n_steps)
    times = torch.arange(n_steps + 1, device=y0.device, dtype=y0.dtype) * dt
    return times, traj


def measured_periods(times: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
    """Time at which each planet's polar angle first reaches 2π (one full orbit), (B,). NaN if never."""
    x, y = traj[:, :, 0, 0], traj[:, :, 0, 1]  # (T, B)
    angle = torch.atan2(y, x)  # (T, B) in (-π, π]
    step = torch.remainder(angle.diff(dim=0) + torch.pi, 2 * torch.pi) - torch.pi  # unwrap jumps
    swept = torch.cat([torch.zeros_like(angle[:1]), step.cumsum(0)])  # (T, B) total angle swept
    periods = torch.full((traj.shape[1],), float("nan"), dtype=times.dtype)
    for b in range(traj.shape[1]):
        k = int(torch.searchsorted(swept[:, b].contiguous(), torch.tensor(2 * torch.pi, dtype=swept.dtype)))
        if 0 < k < len(times):  # linear interpolation between steps k-1 and k
            f = (2 * torch.pi - swept[k - 1, b]) / (swept[k, b] - swept[k - 1, b])
            periods[b] = times[k - 1] + f * (times[k] - times[k - 1])
    return periods
