"""Integrate the geodesic equation with RK4 in proper time τ; measure perihelion advance and infall times."""

import math

import torch

from problems._common.integrators import integrate

from .equation import newton_rhs, rhs


def geodesic(y0: torch.Tensor, M: float, tau_max: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (τ (T,), trajectory (T, B, 2, 4))."""
    n_steps = int(round(tau_max / dt))
    traj = integrate(lambda y, tau: rhs(y, M), y0, dt, n_steps)
    return torch.arange(n_steps + 1, dtype=y0.dtype) * dt, traj


def infall(y0: torch.Tensor, M: float, dt: float, r_stop: float = 2.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Radial plunge from rest until just outside the horizon (r_stop in units of M).

    Schwarzschild coordinates break down at r = 2M (dt/dτ → ∞), so we stop just before it. Returns (τ, trajectory).
    """
    r0 = float(y0[0, 0, 1])
    return geodesic(y0, M, proper_infall_time(r0, r_stop * M, M), dt)


def newton(y0: torch.Tensor, M: float, t_max: float, dt: float) -> torch.Tensor:
    """The Newtonian comparison orbit, (T, B, 2, 2)."""
    return integrate(lambda y, t: newton_rhs(y, M), y0, dt, int(round(t_max / dt)))


def proper_infall_time(r0: float, r: float, M: float) -> float:
    """τ to fall from rest at r0 to r (same cycloid as Newton's radial fall): finite, even to the horizon."""
    eta = math.acos(2 * r / r0 - 1)
    return math.sqrt(r0**3 / (8 * M)) * (eta + math.sin(eta))


def periapsis_angles(traj: torch.Tensor) -> torch.Tensor:
    """φ at each periapsis passage (ṙ changes sign from − to +), linearly interpolated: (K,)."""
    rdot, phi = traj[:, 0, 1, 1], traj[:, 0, 0, 3]
    k = torch.nonzero((rdot[:-1] < 0) & (rdot[1:] >= 0)).flatten()
    f = -rdot[k] / (rdot[k + 1] - rdot[k])
    angles = phi[k] + f * (phi[k + 1] - phi[k])
    return torch.cat([phi[:1], angles])  # the run starts at periapsis
