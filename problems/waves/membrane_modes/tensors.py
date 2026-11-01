"""Build the concrete tensors: the 1D stencil matrix, the grid and the drum mask."""

import torch

SIDE = 1.0  # square side or circle diameter L
C = 1.0  # wave speed on the membrane


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Interior points of [0, L]²: X, Y of shape (n, n) and the spacing h = L / (n + 1). The rim is not stored."""
    h = SIDE / (n + 1)
    s = torch.arange(1, n + 1, device=device, dtype=torch.float64) * h
    X, Y = torch.meshgrid(s, s, indexing="ij")
    return X, Y, h


def second_difference(n: int, h: float, device: torch.device) -> torch.Tensor:
    """D_{ac} of shape (n, n): (δ_{a+1,c} − 2δ_{ac} + δ_{a−1,c}) / h², the clamped rim built in."""
    main = torch.full((n,), -2.0, device=device, dtype=torch.float64)
    side = torch.ones(n - 1, device=device, dtype=torch.float64)
    return (torch.diag(main) + torch.diag(side, 1) + torch.diag(side, -1)) / h**2


def drum_mask(X: torch.Tensor, Y: torch.Tensor, shape: str) -> torch.Tensor:
    """Which grid points belong to the drum, flattened to (n²,) bool (p = a·n + b, the same order as torch.kron)."""
    if shape == "square":
        inside = torch.ones_like(X, dtype=torch.bool)
    else:
        r = SIDE / 2
        inside = (X - r) ** 2 + (Y - r) ** 2 < r**2
    return inside.flatten()
