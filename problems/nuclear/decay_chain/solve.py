"""Solve the Bateman system twice: exactly with the matrix exponential, and with RK4 as a check."""

import torch

from problems._common.integrators import integrate

from .equation import propagator, rhs


def exact(a: torch.Tensor, n0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """N_i(t) = [e^{A t}]_{ij} N_j(0) for all times in one batched matrix_exp: shape (T, M)."""
    return torch.einsum("tij,j->ti", propagator(a, t), n0)


def rk4(a: torch.Tensor, n0: torch.Tensor, t_max: float, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Step dN/dt = A N with classical Runge–Kutta: (times (S+1,), N (S+1, M))."""
    dt = t_max / n_steps
    traj = integrate(lambda n, _t: rhs(n, a), n0, dt, n_steps)
    times = torch.arange(n_steps + 1, dtype=n0.dtype, device=n0.device) * dt
    return times, traj
