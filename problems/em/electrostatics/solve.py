"""Solve L φ = −ρ/ε₀ for the potential, then take E = −∇φ."""

import torch

from .equation import electric_field, laplacian


def solve(rho: torch.Tensor, h: float, eps0: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (φ of shape (N, N), E of shape (2, N, N)).

    The whole PDE is one dense linear system of size N² — fine for N ≲ 80 (6400 unknowns).
    """
    n = rho.shape[0]
    L = laplacian(n, h)  # (N², N²)
    phi = torch.linalg.solve(L, -rho.reshape(-1) / eps0).reshape(n, n)  # (N, N)
    return phi, electric_field(phi, h)


def flux_through_square(E: torch.Tensor, h: float, a0: int, a1: int, b0: int, b1: int) -> torch.Tensor:
    """∮ E·n dl around the box of rows a0..a1 and columns b0..b1 (outward normal n), by the midpoint rule."""
    E_x, E_y = E
    right = E_x[a0 : a1 + 1, b1].sum()
    left = -E_x[a0 : a1 + 1, b0].sum()
    top = E_y[a1, b0 : b1 + 1].sum()
    bottom = -E_y[a0, b0 : b1 + 1].sum()
    return (right + left + top + bottom) * h
