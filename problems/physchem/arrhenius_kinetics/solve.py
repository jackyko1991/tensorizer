"""Integrate all rate laws with RK4, then analyse the "measured" run: which order, which k, which E_a."""

import torch

from problems._common.integrators import integrate

from . import equation
from .tensors import N_SAMPLES

N_STEPS = 400  # RK4 steps in reduced time τ ∈ [0, 1]


def integrate_runs(
    conc0: torch.Tensor, k: torch.Tensor, n: torch.Tensor, t_end: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (τ (N+1,), [A] (N+1, O, T)). All orders and temperatures advance together in one (O, T) tensor."""
    traj = integrate(lambda c, tau: equation.rhs(c, k, n, t_end), conc0, 1.0 / N_STEPS, N_STEPS)
    tau = torch.linspace(0.0, 1.0, N_STEPS + 1, device=conc0.device, dtype=conc0.dtype)
    return tau, traj


def crossing_time(tau: torch.Tensor, traj: torch.Tensor, t_end: torch.Tensor, level: float) -> torch.Tensor:
    """Time at which the (decreasing) [A] first drops to `level`, by linear interpolation.   → (O, T)."""
    i = ((traj > level).sum(0) - 1).clamp(0, len(tau) - 2)  # last grid point still above the level
    c_lo = traj.gather(0, i[None]).squeeze(0)
    c_hi = traj.gather(0, (i + 1)[None]).squeeze(0)
    frac = (c_lo - level) / (c_lo - c_hi)
    return (tau[i] + frac * (tau[1] - tau[0])) * t_end


def sample(tau: torch.Tensor, traj: torch.Tensor, order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick N_SAMPLES evenly spaced points of the run with order `order`.   Returns (τ_s (S,), [A]_{ts} (T, S))."""
    idx = torch.linspace(0, len(tau) - 1, N_SAMPLES, device=tau.device).round().long()
    return tau[idx], traj[idx, order].T


def fit_linearizations(times: torch.Tensor, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight-line fits of [A], ln[A] and 1/[A] against t for every temperature.

    times, data: (T, S). Returns β (3, T, 2) and R² (3, T). The noise σ on [A] becomes σ·|dg/d[A]| on the plotted
    quantity g (σ/[A] for ln[A], σ/[A]² for 1/[A]), so each point is weighted by w = (dg/d[A])⁻² = (1, [A]², [A]⁴):
    late, low-concentration points count less. Points below 2 % of [A]_0 are left out entirely (w = 0).
    """
    g = equation.linearize(data)  # (3, T, S)
    keep = (data > 0.02).to(data.dtype)  # (T, S)
    c = data.clamp_min(0.02)
    w = keep * torch.stack([torch.ones_like(c), c**2, c**4])  # (3, T, S) error-propagation weights
    x = times.expand_as(g)
    beta = equation.least_squares(x, g, w)
    return beta, equation.r_squared(x, g, beta, w)
