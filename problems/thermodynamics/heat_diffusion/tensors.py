"""Build the concrete tensors: the grid, the material constants and the initial hot spots."""

import torch

PLATE = 0.1  # side of the square plate (m)
K = 400.0  # thermal conductivity k (W / (m K)), roughly copper
T_EDGE = 20.0  # temperature of the cold edges (°C)
T_HOT = 100.0  # peak temperature of a hot spot (°C)


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Cell centres x_j = (j + 1) dx, j = 0..n−1, so the ghost cells sit exactly on the plate edges.

    Returns X, Y of shape (n, n) (row index = y, column index = x) and the spacing dx.
    """
    dx = PLATE / (n + 1)
    x = (torch.arange(n, device=device) + 1) * dx  # (n,)
    Y, X = torch.meshgrid(x, x, indexing="ij")  # (n, n) each
    return X, Y, dx


def heat_capacity(alpha: float) -> float:
    """ρc = k / α  (J / (m³ K)), chosen so the diffusivity matches the requested α."""
    return K / alpha


def hot_spots(X: torch.Tensor, Y: torch.Tensor, n_spots: int) -> torch.Tensor:
    """Initial temperature T(y, x): cold plate plus n_spots Gaussian hot spots on a ring.  Shape (1, 1, n, n).

    conv2d wants (batch, channel, height, width), so we keep two leading singleton dims.
    """
    angle = 2 * torch.pi * torch.arange(n_spots) / n_spots + 0.3  # (S,)
    radius = 0.0 if n_spots == 1 else 0.25 * PLATE
    cx = 0.5 * PLATE + radius * torch.cos(angle)  # (S,) spot centres
    cy = 0.5 * PLATE + radius * torch.sin(angle)
    width = 0.1 * PLATE

    r2 = (X[None] - cx[:, None, None].to(X)) ** 2 + (Y[None] - cy[:, None, None].to(X)) ** 2  # (S, n, n)
    bumps = torch.exp(-r2 / (2 * width**2)).amax(dim=0)  # (n, n) highest bump at each cell, ≤ 1
    T = T_EDGE + (T_HOT - T_EDGE) * bumps
    return T[None, None]


def sine_mode(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """T = sin(πx/L) sin(πy/L): the slowest-decaying mode with T = 0 on the edges.  Shape (1, 1, n, n)."""
    return (torch.sin(torch.pi * X / PLATE) * torch.sin(torch.pi * Y / PLATE))[None, None]
