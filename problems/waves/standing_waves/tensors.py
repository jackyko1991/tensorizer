"""Build the concrete tensors: the grid, the stencil matrix and the starting shapes of the strings."""

import math

import torch


def grid(L: float, n_points: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """x of shape (N,) from 0 to L, ends included, and the spacing Δx."""
    x = torch.linspace(0.0, L, n_points, device=device)
    return x, L / (n_points - 1)


def stencil(m: int, dx: float, device: torch.device) -> torch.Tensor:
    """D_{ij} of shape (M, M): (δ_{i+1,j} − 2δ_{ij} + δ_{i−1,j}) / Δx².

    The fixed ends u = 0 are built in: the first and last rows simply miss one neighbour.
    """
    main = -2.0 * torch.ones(m, device=device)
    side = torch.ones(m - 1, device=device)
    return (torch.diag(main) + torch.diag(side, 1) + torch.diag(side, -1)) / dx**2


def initial_shapes(x: torch.Tensor, L: float, n_harmonics: int, pluck_at: float) -> torch.Tensor:
    """u0 of shape (B, N) with B = n_harmonics + 1.

    rows 0 … H−1:  sin(nπx/L) for n = 1 … H     (single harmonics)
    row  H:        a triangle with its peak at x_p = pluck_at · L    (plucked string)
    """
    n = torch.arange(1, n_harmonics + 1, device=x.device, dtype=x.dtype)  # (H,)
    harmonics = torch.sin(math.pi * n[:, None] * x[None, :] / L)  # (H, N)
    xp = pluck_at * L
    pluck = torch.where(x < xp, x / xp, (L - x) / (L - xp))  # (N,) height 1 at x_p, 0 at both ends
    return torch.cat([harmonics, pluck[None, :]], dim=0)
