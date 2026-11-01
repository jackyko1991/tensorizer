"""Build the concrete tensors: the initial generalised coordinates and velocities.

float64 so that energy conservation is limited by the integrator, not by rounding.
"""

import math

import torch

DTYPE = torch.float64


def initial_state(r0: float, theta0_deg: float, device: torch.device | None = None) -> torch.Tensor:
    """y0 of shape (2, 2): released from rest.

    y0[0] = (r, θ)         generalised coordinates q_i
    y0[1] = (0, 0)         generalised velocities q̇_i
    """
    q0 = torch.tensor([r0, math.radians(theta0_deg)], dtype=DTYPE, device=device)  # (2,)
    return torch.stack([q0, torch.zeros_like(q0)])
