"""Integrate the autograd-derived equations of motion with 4th-order Runge–Kutta."""

import torch

from problems._common.integrators import integrate

from .equation import Lagrangian, energy, rhs


def solve(y0: torch.Tensor, L: Lagrangian, t_max: float, dt: float = 0.01) -> dict[str, torch.Tensor]:
    """Return times (T,), coordinates q (T, n), velocities q̇ (T, n) and the energy E (T,).

    Every RK4 stage asks autograd for ∂L/∂q, H and C at the current state: no equation was typed in by hand.
    """
    n_steps = int(round(t_max / dt))
    traj = integrate(lambda y, t: rhs(y, L), y0, dt, n_steps)  # (T, 2, n)
    q, q_dot = traj[:, 0], traj[:, 1]
    E = energy(L, q, q_dot)  # (T,) energy at every time, in one batched call
    times = torch.arange(n_steps + 1, device=y0.device, dtype=y0.dtype) * dt
    return {"times": times, "q": q, "q_dot": q_dot, "E": E}
