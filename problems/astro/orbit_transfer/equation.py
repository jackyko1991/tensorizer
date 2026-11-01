r"""Physics → tensor translation.

A spacecraft coasting around a central body of gravitational parameter μ = GM feels only its gravity:

    \ddot{\mathbf r} = -\frac{\mu}{r^3}\,\mathbf r

In index notation for a batch of spacecraft b and components i ∈ {x, y}:

    \ddot x_{bi} = -\mu \,x_{bi}\,(x_{bj} x_{bj})^{-3/2}

A repeated index (j above) is summed — that is einsum('bj,bj->b'). Between burns the energy per unit mass

    E_b = ½ v_{bi} v_{bi} - μ / \sqrt{x_{bj} x_{bj}} = -μ / (2 a_b)

is constant. Solving it for the speed gives the vis-viva equation v² = μ (2/r - 1/a): the speed anywhere on an orbit
of semi-major axis a. Every Δv below is a difference of two vis-viva speeds at the same radius.

An impulsive burn changes the velocity instantly along its own direction (prograde for Δv > 0, retrograde for Δv < 0):

    v_{bi} \leftarrow v_{bi} + Δv_b \, v_{bi} / \sqrt{v_{bj} v_{bj}}
"""

import math

import torch


def acceleration(x: torch.Tensor, mu: float) -> torch.Tensor:
    r"""\ddot x_{bi} = -μ x_{bi} (x_{bj} x_{bj})^{-3/2}.   x: (B, 2) → (B, 2)"""
    r2 = torch.einsum("bj,bj->b", x, x)  # (B,)  squared distance to the centre
    return -mu * torch.einsum("b,bi->bi", r2**-1.5, x)


def rhs(y: torch.Tensor, mu: float) -> torch.Tensor:
    """dy/dt for the state y of shape (B, 2, 2): y[:, 0] = position, y[:, 1] = velocity."""
    return torch.stack([y[:, 1], acceleration(y[:, 0], mu)], dim=1)


def energy(y: torch.Tensor, mu: float) -> torch.Tensor:
    r"""E_b = ½ v_{bi} v_{bi} - μ / r_b.   y: (..., 2, 2) → (...)"""
    x, v = y[..., 0, :], y[..., 1, :]
    return 0.5 * torch.einsum("...i,...i->...", v, v) - mu / torch.einsum("...i,...i->...", x, x).sqrt()


def apply_burn(y: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
    r"""v_{bi} ← v_{bi} + Δv_b v_{bi} / |v_b|: an instant burn along the velocity.   y: (B, 2, 2), dv: (B,)"""
    x, v = y[:, 0], y[:, 1]
    speed = torch.einsum("bj,bj->b", v, v).sqrt()  # (B,)
    v_new = v + torch.einsum("b,bi->bi", dv / speed, v)
    return torch.stack([x, v_new], dim=1)


# --- closed-form transfer costs (all broadcast over tensors of radii) ---------------------------------


def vis_viva(r: torch.Tensor, a: torch.Tensor, mu: float) -> torch.Tensor:
    r"""Speed at radius r on an orbit of semi-major axis a:  v = \sqrt{μ (2/r - 1/a)}."""
    return torch.sqrt(mu * (2 / r - 1 / a))


def circular_speed(r: torch.Tensor, mu: float) -> torch.Tensor:
    """Vis-viva with a = r."""
    return torch.sqrt(mu / r)


def half_period(a: torch.Tensor, mu: float) -> torch.Tensor:
    r"""Time to fly half an ellipse, periapsis → apoapsis:  t = π \sqrt{a^3 / μ}."""
    return math.pi * torch.sqrt(a**3 / mu)


def hohmann(r1: torch.Tensor, r2: torch.Tensor, mu: float):
    """Two burns via the ellipse that touches both circles. Returns signed (Δv₁, Δv₂) and the transfer time."""
    a = (r1 + r2) / 2  # transfer ellipse: periapsis r1, apoapsis r2 (or the other way round)
    dv1 = vis_viva(r1, a, mu) - circular_speed(r1, mu)  # leave the inner circle
    dv2 = circular_speed(r2, mu) - vis_viva(r2, a, mu)  # circularise on arrival
    return dv1, dv2, half_period(a, mu)


def bi_elliptic(r1: torch.Tensor, r2: torch.Tensor, rb: torch.Tensor, mu: float):
    """Three burns via two half-ellipses that meet far out at r_b. Returns signed (Δv₁, Δv₂, Δv₃) and the time."""
    a1 = (r1 + rb) / 2  # first ellipse: r1 → r_b
    a2 = (rb + r2) / 2  # second ellipse: r_b → r2
    dv1 = vis_viva(r1, a1, mu) - circular_speed(r1, mu)  # out to r_b
    dv2 = vis_viva(rb, a2, mu) - vis_viva(rb, a1, mu)  # at r_b: raise the far side to r2
    dv3 = circular_speed(r2, mu) - vis_viva(r2, a2, mu)  # circularise (a braking burn)
    return dv1, dv2, dv3, half_period(a1, mu) + half_period(a2, mu)


def bi_parabolic(r1: torch.Tensor, r2: torch.Tensor, mu: float) -> torch.Tensor:
    r"""Total Δv of the bi-elliptic transfer in the limit r_b → ∞:  (\sqrt2 - 1)(v_{c1} + v_{c2})."""
    return (math.sqrt(2) - 1) * (circular_speed(r1, mu) + circular_speed(r2, mu))
