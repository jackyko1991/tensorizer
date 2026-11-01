"""Build the concrete tensors: initial 4-positions and 4-velocities for a bound orbit and a radial infall.

Units: G = c = 1 and M = 1, so lengths are in GM/c² (1.5 km for one solar mass) and the horizon is at r = 2.
Orbits lie in the equatorial plane θ = π/2.
"""

import math

import torch

M = 1.0


def orbit_constants(p: float, e: float) -> tuple[float, float]:
    """Energy E and angular momentum L per unit mass of the Schwarzschild orbit with semi-latus rectum p and
    eccentricity e (turning points r = p/(1 ± e)). Needs p > 6 + 2e for a stable bound orbit."""
    L = math.sqrt(p**2 * M / (p - 3 * M - e**2 * M))
    E = math.sqrt(((p - 2 * M) ** 2 - 4 * e**2 * M**2) / (p * (p - 3 * M - e**2 * M)))
    return E, L


def bound_orbit(p: float, e: float) -> torch.Tensor:
    """y0 of shape (1, 2, 4): at periapsis r = p/(1+e), moving purely in φ.

    u^μ = (E / (1 − 2M/r), 0, 0, L / r²) — the conserved E and L fix the 4-velocity.
    """
    E, L = orbit_constants(p, e)
    r = p / (1 + e)
    x = torch.tensor([0.0, r, math.pi / 2, 0.0], dtype=torch.float64)
    u = torch.tensor([E / (1 - 2 * M / r), 0.0, 0.0, L / r**2], dtype=torch.float64)
    return torch.stack([x, u])[None]


def radial_infall(r0: float) -> torch.Tensor:
    """y0 of shape (1, 2, 4): released from rest at r0, so u^μ = (1/\\sqrt{1 − 2M/r0}, 0, 0, 0)."""
    x = torch.tensor([0.0, r0, math.pi / 2, 0.0], dtype=torch.float64)
    u = torch.tensor([1 / math.sqrt(1 - 2 * M / r0), 0.0, 0.0, 0.0], dtype=torch.float64)
    return torch.stack([x, u])[None]


def newton_orbit(p: float, e: float) -> torch.Tensor:
    """y0 of shape (1, 2, 2): the Newtonian ellipse with the same p and e, at periapsis (vis-viva speed)."""
    r = p / (1 + e)
    v = math.sqrt(M * (1 + e) / r)
    return torch.tensor([[[r, 0.0], [0.0, v]]], dtype=torch.float64)
