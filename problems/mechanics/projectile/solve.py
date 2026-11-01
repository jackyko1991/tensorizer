"""Integrate the batched ODE with 4th-order Runge–Kutta and stop each ball at the ground."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def solve(
    y0: torch.Tensor, g: torch.Tensor, k: float, t_max: float, dt: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), trajectory (T, B, 2, 2)).

    All B balls advance together: one RK4 step updates the whole (B, 2, 2) tensor.
    After a ball hits the ground (z < 0) we freeze it at its landing point.
    """
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda y, t: rhs(y, g, k), y0, dt, n_steps)  # (T, B, 2, 2)

    z = traj[:, :, 0, 1]  # (T, B) height of each ball
    landed = torch.cummax((z < 0).int(), dim=0).values.bool()  # stays True after first touchdown
    landed[0] = False
    step_of_landing = torch.where(landed.any(0), landed.int().argmax(0), len(z) - 1)  # (B,)
    landing = traj[step_of_landing, torch.arange(traj.shape[1])].clone()  # (B, 2, 2)
    landing[:, 0, 1] = 0.0  # on the ground
    landing[:, 1] = 0.0  # at rest
    traj = torch.where(landed[:, :, None, None], landing, traj)

    times = torch.arange(n_steps + 1, device=y0.device) * dt
    return times, traj


def ranges(traj: torch.Tensor) -> torch.Tensor:
    """Horizontal distance travelled by each ball at the end of the run: (B,)."""
    return traj[-1, :, 0, 0]
