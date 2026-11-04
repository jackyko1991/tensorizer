"""Build the concrete tensors: a random flock in a periodic box, the noise levels and the Reynolds weights."""

import math

import torch

# Reynolds weights per unit time for v0 = 0.05 (neighbour radius R = 1, about 20 neighbours at the default density).
# Strong cohesion makes the flock collapse into a milling ball, so cohesion is kept weak.
REYNOLDS = {"separation": 0.0003, "alignment": 0.02, "cohesion": 0.0001}


def reynolds_weights(speed: float) -> dict:
    """Separation and cohesion steer relative to the speed, so they scale with v0 / 0.05."""
    return {k: w * (speed / 0.05 if k != "alignment" else 1.0) for k, w in REYNOLDS.items()}


def random_flock(batch: int, n: int, box: float, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform positions in [0, L)² and uniform headings, independent for every batch member:
    positions (B, N, 2), headings θ (B, N)."""
    gen = torch.Generator().manual_seed(seed)
    x = box * torch.rand(batch, n, 2, generator=gen)
    theta = 2 * math.pi * torch.rand(batch, n, generator=gen) - math.pi
    return x.to(device), theta.to(device)


def noise_levels(count: int, device: torch.device) -> torch.Tensor:
    """count noise amplitudes η from 0 to 2π (2π = completely random headings): (B, 1)."""
    return torch.linspace(0, 2 * math.pi, count, device=device)[:, None]


def velocities(theta: torch.Tensor, speed: float) -> torch.Tensor:
    """v = v0 (cos θ, sin θ).   (B, N) → (B, N, 2)"""
    return speed * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
