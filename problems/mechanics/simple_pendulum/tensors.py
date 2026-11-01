"""Build the concrete tensors: the release amplitudes and the batched initial state."""

import torch


def amplitudes(n: int, max_deg: float) -> torch.Tensor:
    """n release angles from 10° up to max_deg (or just max_deg for a single pendulum), in radians: (B,)."""
    if n == 1:
        return torch.deg2rad(torch.tensor([max_deg]))
    return torch.deg2rad(torch.linspace(10.0, max_deg, n))


def initial_state(theta0: torch.Tensor, device: torch.device) -> torch.Tensor:
    """y0 of shape (B, 2): every pendulum is released from rest.

    y0[b, 0] = θ_b(0)     release angle
    y0[b, 1] = 0          angular velocity ω_b(0)
    """
    theta0 = theta0.to(device)
    return torch.stack([theta0, torch.zeros_like(theta0)], dim=1)
