"""Build the concrete tensors: first guesses for the five Lagrange points and the test particles near L4."""

import math

import torch


def lagrange_guesses(mu: float, device: torch.device) -> torch.Tensor:
    """Rough positions (5, 3) of L1 … L5 (Hill-sphere estimates for L1/L2, triangles for L4/L5)."""
    h = (mu / 3) ** (1 / 3)
    s = math.sqrt(3) / 2
    return torch.tensor(
        [
            [1 - mu - h, 0.0, 0.0],  # L1 between the bodies
            [1 - mu + h, 0.0, 0.0],  # L2 beyond the planet
            [-1 - 5 * mu / 12, 0.0, 0.0],  # L3 opposite the planet
            [0.5 - mu, s, 0.0],  # L4 leading by 60°
            [0.5 - mu, -s, 0.0],  # L5 trailing by 60°
        ],
        dtype=torch.float64,
        device=device,
    )


def particles_near_l4(mu: float, nudges_deg: list[float], device: torch.device) -> torch.Tensor:
    """Initial states (B, 2, 3): particles at rest in the rotating frame on the unit circle around the star,
    nudged ahead of L4 (which sits 60° ahead of the planet) by the given angles."""
    theta = torch.deg2rad(60.0 + torch.tensor(nudges_deg, dtype=torch.float64, device=device))  # (B,)
    x = torch.stack([-mu + torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)  # (B, 3)
    return torch.stack([x, torch.zeros_like(x)], dim=1)


def potential_grid(n: int, reach: float, device: torch.device) -> torch.Tensor:
    """Points (n, n, 3) of the z = 0 plane for the contour plot."""
    s = torch.linspace(-reach, reach, n, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(s, s, indexing="xy")
    return torch.stack([X, Y, torch.zeros_like(X)], dim=-1)
