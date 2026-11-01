"""Build the concrete tensors: the stack of layers and the fan of rays leaving the source."""

import torch

THICKNESS = 1.0  # every layer is 1 unit thick: layer k fills  -(k+1) < y < -k
WIDTH = 5.0  # the box spans 0 < x < WIDTH


def indices(n1: float, n2: float, n3: float, device: torch.device) -> torch.Tensor:
    """Refractive index of each layer, top to bottom: shape (L,) = (3,)."""
    return torch.tensor([n1, n2, n3], device=device)


def fan_angles(n_rays: int, max_angle_deg: float) -> torch.Tensor:
    """Angles from the vertical, spread from 5° to max_angle, in radians: shape (R,)."""
    return torch.deg2rad(torch.linspace(5.0, max_angle_deg, n_rays))


def initial_rays(angles: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Every ray starts at the top-left corner and heads down-right.

    p[r, :] = (0, 0)                        position p_{ri}          (R, 2)
    d[r, :] = (sin θ_r, −cos θ_r)           unit direction d_{ri}    (R, 2)
    """
    angles = angles.to(device)
    d = torch.stack([torch.sin(angles), -torch.cos(angles)], dim=-1)  # (R, 2)
    p = torch.zeros_like(d)
    return p, d
