"""Integrate all rockets at once with RK4; a rocket that falls back is frozen where it lands."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def solve(y0: torch.Tensor, GM: float, R: float, t_max: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, B, 2, 2))."""
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda y, t: rhs(y, GM), y0, dt, n_steps)

    r = traj[:, :, 0, :].norm(dim=-1)  # (T, B) distance from the centre
    below = r < R * (1 - 1e-4)  # small margin: at launch r = R up to round-off
    landed = torch.cummax(below.int(), dim=0).values.bool()  # stays True after touchdown
    step = torch.where(landed.any(0), landed.int().argmax(0), len(r) - 1)  # (B,)
    rest = traj[step, torch.arange(traj.shape[1])].clone()  # (B, 2, 2)
    rest[:, 0] = R * rest[:, 0] / rest[:, 0].norm(dim=-1, keepdim=True)  # back on the surface
    rest[:, 1] = 0.0
    traj = torch.where(landed[:, :, None, None], rest, traj)

    times = torch.arange(n_steps + 1, device=y0.device, dtype=y0.dtype) * dt
    return times, traj


def escaped(traj: torch.Tensor, R: float) -> torch.Tensor:
    """True for rockets still in flight at the end and moving away from the planet, (B,)."""
    x, v = traj[-1, :, 0], traj[-1, :, 1]
    return (x.norm(dim=-1) > 1.01 * R) & (torch.einsum("bi,bi->b", x, v) > 0)


def highest_point(traj: torch.Tensor) -> torch.Tensor:
    """Largest distance from the centre reached by each rocket, (B,)."""
    return traj[:, :, 0, :].norm(dim=-1).max(dim=0).values
