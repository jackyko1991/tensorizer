"""Build the concrete tensors: two almost identical initial states, stacked in a batch.

We use float64: chaos amplifies every rounding error, so we want those errors as small as possible.
"""

import math

import torch

DTYPE = torch.float64


def initial_state(theta1_deg: float, theta2_deg: float, delta_deg: float, device: torch.device) -> torch.Tensor:
    """y0 of shape (B=2, 2, 2): both pendulums start at rest; the second has θ_2 nudged by delta.

    y0[b, 0, :] = (θ_1, θ_2)     angles q_bi (radians, measured from the downward vertical)
    y0[b, 1, :] = (0, 0)         angular velocities q̇_bi
    """
    angles = torch.tensor(
        [
            [math.radians(theta1_deg), math.radians(theta2_deg)],
            [math.radians(theta1_deg), math.radians(theta2_deg + delta_deg)],
        ],
        dtype=DTYPE,
        device=device,
    )  # (2, 2)
    return torch.stack([angles, torch.zeros_like(angles)], dim=1)
