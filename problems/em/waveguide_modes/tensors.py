"""Build the concrete tensors: the grids for TM (nodes inside the walls) and TE (cell centres)."""

import torch


def grid_size(a: float, b: float, n_b: int) -> tuple[int, int]:
    """(Nx, Ny): n_b points across the short side b, and the same spacing along the long side a."""
    return max(4, round(n_b * a / b)), n_b


def grid(a: float, b: float, nx: int, ny: int, kind: str) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Return (x (Nx,), y (Ny,), hx, hy).

    TM (Dirichlet): Nx interior nodes, h = a/(Nx + 1), so the walls x = 0 and x = a are the missing nodes.
    TE (Neumann):   Nx cells of width h = a/Nx, the values sit at the cell centres.
    """
    if kind == "TM":
        hx, hy = a / (nx + 1), b / (ny + 1)
        x = torch.arange(1, nx + 1, dtype=torch.float64) * hx
        y = torch.arange(1, ny + 1, dtype=torch.float64) * hy
    else:
        hx, hy = a / nx, b / ny
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * hx
        y = (torch.arange(ny, dtype=torch.float64) + 0.5) * hy
    return x, y, hx, hy


def candidate_labels(kind: str, max_index: int = 8) -> list[tuple[int, int]]:
    """The (m, n) pairs a mode of this kind can have: TM needs m, n ≥ 1; TE allows 0 but not both."""
    lowest = 1 if kind == "TM" else 0
    pairs = [(m, n) for m in range(lowest, max_index) for n in range(lowest, max_index)]
    return [p for p in pairs if p != (0, 0)]
