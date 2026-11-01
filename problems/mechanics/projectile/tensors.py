"""Build the concrete tensors: gravity vector and the batched initial state."""

import math

import torch


def gravity(device: torch.device) -> torch.Tensor:
    """g_i with i ∈ {x, z}: shape (2,)."""
    return torch.tensor([0.0, -9.81], device=device)


def launch_angles(n: int) -> torch.Tensor:
    """n angles spread between 15° and 75° (or just 45° for a single ball), in radians."""
    if n == 1:
        return torch.tensor([math.radians(45.0)])
    return torch.deg2rad(torch.linspace(15.0, 75.0, n))


def initial_state(v0: float, angles: torch.Tensor, device: torch.device) -> torch.Tensor:
    """y0 of shape (B, 2, 2): every ball starts at the origin with speed v0 at its own angle.

    y0[b, 0, :] = (0, 0)                            position x_{bi}
    y0[b, 1, :] = v0 (cos θ_b, sin θ_b)             velocity v_{bi}
    """
    angles = angles.to(device)
    velocity = v0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (B, 2)
    position = torch.zeros_like(velocity)
    return torch.stack([position, velocity], dim=1)
