"""Integrate the whole batch of trajectories with RK4 and fit the growth rate of their spread."""

import math

import torch

from problems._common.integrators import integrate

from .equation import rhs


def settle(A: torch.Tensor, Q: torch.Tensor, t: float = 10.0, dt: float = 0.005) -> torch.Tensor:
    """Start at (1, 1, 1) and run for a while, so the ball starts on the attractor.  → (3,)"""
    x0 = torch.ones(1, 3, dtype=A.dtype, device=A.device)  # (B=1, 3)
    traj = integrate(lambda x, _t: rhs(x, A, Q), x0, dt, int(round(t / dt)))
    return traj[-1, 0]


def solve(x0: torch.Tensor, A: torch.Tensor, Q: torch.Tensor, t_max: float, dt: float = 0.005):
    """Return times (T,) and trajectories (T, B, 3).  x0: (B, 3)"""
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda x, _t: rhs(x, A, Q), x0, dt, n_steps)
    times = torch.arange(n_steps + 1, dtype=x0.dtype, device=x0.device) * dt
    return times, traj


def lyapunov_fit(times: torch.Tensor, s: torch.Tensor, high: float = 1.0):
    r"""Fit ln s = c + λ t by least squares while the spread grows from 10 s(0) to `high`.

    Past `high` the trajectories are no longer close: the spread saturates at the size of the attractor.
    Returns (λ, t_start, t_end, c); λ is nan if the spread never grows through the window.
    """
    above_low, above_high = (s > 10 * s[0]).nonzero(), (s > high).nonzero()
    if len(above_low) == 0 or len(above_high) == 0 or above_high[0] - above_low[0] < 3:
        return math.nan, math.nan, math.nan, math.nan
    i0, i1 = above_low[0].item(), above_high[0].item()
    t, y = times[i0 : i1 + 1], s[i0 : i1 + 1].log()
    X = torch.stack([torch.ones_like(t), t], dim=1)  # (n, 2) least squares for [c, λ]
    c, lam = torch.linalg.lstsq(X, y[:, None]).solution[:, 0].tolist()
    return lam, t[0].item(), t[-1].item(), c
