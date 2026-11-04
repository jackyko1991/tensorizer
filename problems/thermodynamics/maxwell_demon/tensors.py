"""Concrete tensors: units, the box, the soft walls, and equilibrium samples.

Units: kT = 1, mobility = 1 (so D = 1) and the half-box width a = 1. The box is [0, 2a];
the demon's partition sits at x = p; every wall is a stiff spring κu²/2 on the penetration u > 0.
"""

import torch

A = 1.0  # half-box width a: one bit = "left half" or "right half"
KAPPA = 500.0  # wall stiffness (kT / a²): walls are soft over √(π/2κ) ≈ 0.06 a
DT = 8e-4  # time step (a² / D); κ·dt = 0.4, corrected exactly by a Metropolis test (MALA)
DTYPE = torch.float64


def x_grid(device: torch.device, n: int = 8001) -> torch.Tensor:
    """(X,) quadrature grid that covers the box and the soft walls."""
    return torch.linspace(-0.4, 2 * A + 0.4, n, dtype=DTYPE, device=device)


def full(n: int, value: float, device: torch.device) -> torch.Tensor:
    return torch.full((n,), float(value), dtype=DTYPE, device=device)
