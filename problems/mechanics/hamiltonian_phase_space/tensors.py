"""Build the concrete tensors: a fan of orbits, the integrator test orbit and the Liouville blob.

Every phase-space point is a row z = (q, p); float64 so that energy errors come from the integrator, not rounding.
"""

import math

import torch

DTYPE = torch.float64


def orbit_fan(n: int, p_sep: float, device=None) -> torch.Tensor:
    """n orbits started at q = 0 with momenta up to 1.5 × the separatrix momentum: (n, 2).

    |p| < p_sep ⇒ libration (swinging), |p| > p_sep ⇒ rotation (going over the top).
    """
    p0 = torch.linspace(0.15, 1.5, n, dtype=DTYPE, device=device) * p_sep  # (n,)
    p0 = p0 * torch.where(torch.arange(n, device=device) % 2 == 0, 1.0, -1.0)  # alternate the direction
    return torch.stack([torch.zeros_like(p0), p0], dim=1)


def test_orbit(amplitude_deg: float, device=None) -> torch.Tensor:
    """One pendulum released from rest at the given angle: (1, 2)."""
    return torch.tensor([[math.radians(amplitude_deg), 0.0]], dtype=DTYPE, device=device)


def blob(center: tuple[float, float], radius: tuple[float, float], k: int = 2000, device=None) -> torch.Tensor:
    """K points on the boundary of a small ellipse in the (q, p) plane, counter-clockwise: (K, 2).

    Many points, because the boundary stretches into a long thin spiral and the polygon must still follow it.
    """
    phi = torch.linspace(0, 2 * math.pi, k + 1, dtype=DTYPE, device=device)[:-1]  # (K,)
    q = center[0] + radius[0] * torch.cos(phi)
    p = center[1] + radius[1] * torch.sin(phi)
    return torch.stack([q, p], dim=1)
