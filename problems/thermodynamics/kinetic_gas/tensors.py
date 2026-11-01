"""Build the concrete tensors: positions on a jittered grid and velocities of equal speed.

Units are "reduced": box side L = 1, particle mass m = 1 and Boltzmann's constant k = 1.
"""

import math

import torch

BOX = 1.0  # side L of the square box
MASS = 1.0


def initial_positions(n: int, radius: float, device: torch.device, seed: int = 0) -> torch.Tensor:
    """x_{ai}: n disks on a square grid, each nudged randomly inside its own cell (no overlaps).  (N, 2)"""
    gen = torch.Generator().manual_seed(seed)
    per_side = math.ceil(math.sqrt(n))
    cell = BOX / per_side
    i, j = torch.meshgrid(torch.arange(per_side), torch.arange(per_side), indexing="ij")
    centres = (torch.stack([i, j], dim=-1).reshape(-1, 2)[:n] + 0.5) * cell  # (N, 2) cell centres
    wiggle = 0.5 * cell - radius  # how far a disk may move and still stay in its cell
    jitter = (2 * torch.rand(n, 2, generator=gen) - 1) * 0.9 * wiggle
    return (centres + jitter).to(device)


def initial_velocities(n: int, speed: float, device: torch.device, seed: int = 1) -> torch.Tensor:
    """v_{ai}: every disk has the same speed, random direction.  (N, 2)

    The second half moves exactly opposite to the first half, so the total momentum is zero.
    """
    gen = torch.Generator().manual_seed(seed)
    angle = 2 * math.pi * torch.rand(n, generator=gen)  # (N,)
    half = n // 2
    angle[half : 2 * half] = angle[:half] + math.pi
    return (speed * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)).to(device)
