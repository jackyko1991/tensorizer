"""Build the concrete tensors: gravity, the ramp angles and each ramp's unit vectors."""

import torch


def gravity(g: float, device: torch.device) -> torch.Tensor:
    """g_i with i ∈ {x, z}: shape (2,)."""
    return torch.tensor([0.0, -g], device=device)


def ramp_angles(n: int, min_deg: float, max_deg: float) -> torch.Tensor:
    """n incline angles from min_deg to max_deg (or just max_deg for a single ramp), in radians: (B,)."""
    if n == 1:
        return torch.deg2rad(torch.tensor([max_deg]))
    return torch.deg2rad(torch.linspace(min_deg, max_deg, n))


def unit_vectors(theta: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Down-slope s_bi and surface normal n_bi for every ramp: two tensors of shape (B, 2).

    s[b] = ( cos θ_b, -sin θ_b)
    n[b] = ( sin θ_b,  cos θ_b)
    """
    theta = theta.to(device)
    c, s = torch.cos(theta), torch.sin(theta)
    down_slope = torch.stack([c, -s], dim=-1)
    normal = torch.stack([s, c], dim=-1)
    return down_slope, normal
