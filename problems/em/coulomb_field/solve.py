"""Evaluate the superposed field for every position of the moving charge."""

import torch

from .equation import electric_field


def solve(points: torch.Tensor, path: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
    """Return E of shape (T, G, 2): the field at every point for each of the T charge positions.

    Each time step is one call to electric_field — all G points and both charges at once.
    """
    return torch.stack([electric_field(points, positions, charges) for positions in path])


def each_charge(points: torch.Tensor, path: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
    """Return the field of every charge on its own, shape (T, Q, G, 2).

    Summing over Q gives back solve(...): that is the superposition principle.
    """
    alone = [solve(points, path[:, q : q + 1], charges[q : q + 1]) for q in range(len(charges))]
    return torch.stack(alone, dim=1)
