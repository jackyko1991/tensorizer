"""Build the concrete tensors: the membrane grid, the bleach spot, the stencil and the diffusion coefficients.

Units: lengths in µm, times in s, D in µm²/s. The patch is scaled with the bleach radius w (12 w wide,
8 grid points per w), so the spot is always drawn with the same resolution.
"""

import torch

POINTS_PER_RADIUS = 8  # grid spacing h = w / 8
PATCH = 12  # patch width in units of w; large enough that the bleached area is only ~2 % of it

# name, D (µm²/s): typical orders of magnitude for molecules in a cell membrane
SPECIES = [
    ("lipid", 1.0),
    ("membrane protein", 0.25),
    ("crowded protein", 0.05),
]

# 5-point stencil of the Laplacian: offsets (p_k, q_k) and weights w_k
OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
WEIGHTS = [-4.0, 1.0, 1.0, 1.0, 1.0]


def species_names() -> list[str]:
    return [name for name, _ in SPECIES]


def diffusion_coefficients(device: torch.device) -> torch.Tensor:
    """D_b in µm²/s, shape (B,)."""
    return torch.tensor([D for _, D in SPECIES], device=device)


def grid(w: float, device: torch.device) -> tuple[torch.Tensor, float]:
    """Cell-centre coordinates x_i (N,) in µm, centred on the bleach spot, and the spacing h."""
    n = PATCH * POINTS_PER_RADIUS
    h = w / POINTS_PER_RADIUS
    x = (torch.arange(n, device=device) - (n - 1) / 2) * h
    return x, h


def bleach_mask(x: torch.Tensor, w: float) -> torch.Tensor:
    """M_ij = 1 inside the bleached disk x² + y² ≤ w², else 0.   (N, N)"""
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    return (r2 <= w**2).to(x.dtype)


def stencil(device: torch.device) -> torch.Tensor:
    """Stencil weights w_k, shape (5,)."""
    return torch.tensor(WEIGHTS, device=device)


def initial_state(mask: torch.Tensor, depth: float, n_species: int) -> torch.Tensor:
    """c0 of shape (B, N, N): fluorescence 1 everywhere, reduced to `depth` inside the spot by the laser flash."""
    c = 1.0 - (1.0 - depth) * mask  # (N, N)
    return c.expand(n_species, -1, -1).clone()
