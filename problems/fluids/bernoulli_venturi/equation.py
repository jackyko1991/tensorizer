r"""Physics → tensor translation.

Steady, incompressible, frictionless flow through a pipe whose cross-section A(x) changes slowly.
Treating the flow as one-dimensional (velocity u(x) uniform over each cross-section):

    continuity (mass in = mass out):   A(x) u(x) = Q                       (Q = volume flow rate)
    Bernoulli along a streamline:      p(x) + ½ ρ u(x)^2 = p_1 + ½ ρ u_1^2

Every quantity is a tensor over the axis points x: shape (N,). Tracer particles simply ride
the local velocity, dx_n/dt = u(x_n), for a batch of n tracers at once.
"""

import math

import torch


def area(r: torch.Tensor) -> torch.Tensor:
    """Circular cross-section A = π r².   r: (N,) → (N,)"""
    return math.pi * r**2


def velocity(Q: float, A: torch.Tensor) -> torch.Tensor:
    """Continuity: u = Q / A.   (N,) → (N,)"""
    return Q / A


def pressure(u: torch.Tensor, p1: float, u1: float, rho: float) -> torch.Tensor:
    r"""Bernoulli: p = p_1 + ½ ρ (u_1² − u²).   (N,) → (N,)"""
    return p1 + 0.5 * rho * (u1**2 - u**2)


def venturi_flow_rate(dp: float, A1: float, A2: float, rho: float) -> float:
    r"""Invert the two laws (what a Venturi meter does): Q = A_2 \sqrt{2Δp / (ρ (1 − (A_2/A_1)^2))}."""
    return A2 * math.sqrt(2 * dp / (rho * (1 - (A2 / A1) ** 2)))
