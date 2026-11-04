"""Integrate the effective LQC equations with RK4, a batch of universes in one tensor."""

import torch

from problems._common.integrators import integrate

from .equation import rhs


def evolve(y0: torch.Tensor, p_phi: torch.Tensor, lam: torch.Tensor, dt: float, t_span: float) -> dict:
    """y0 (B, 3) → {"t": (T,), "y": (T, B, 3)} over proper time [0, t_span]."""
    n = int(round(t_span / dt))
    y = integrate(lambda state, t: rhs(state, p_phi, lam), y0, dt, n)
    t = torch.arange(n + 1, dtype=y0.dtype, device=y0.device) * dt
    return {"t": t, "y": y}
