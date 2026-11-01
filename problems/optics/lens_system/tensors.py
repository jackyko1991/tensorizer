"""Build the concrete tensors: the fan of rays from the object tip, the axis samples and the lens gaps."""

import torch


def object_rays(height: float, half_angle: float, n_rays: int) -> torch.Tensor:
    """Rays leaving the tip of the object (height h0) at angles in [-half_angle, half_angle].

    rays[r, :] = (h0, θ_r)          ray vector y_{rj}, j ∈ {height, angle}     (R, 2)
    """
    angles = torch.linspace(-half_angle, half_angle, n_rays)  # (R,)
    return torch.stack([torch.full_like(angles, height), angles], dim=-1)


def axis(z_end: float, n_points: int = 300) -> torch.Tensor:
    """Positions along the optical axis where ray heights are evaluated: (Z,)."""
    return torch.linspace(0.0, z_end, n_points)


def separations(d_start: float, d_end: float, n_frames: int = 120) -> torch.Tensor:
    """Lens gap for every animation frame: (F,)."""
    return torch.linspace(d_start, d_end, n_frames)
