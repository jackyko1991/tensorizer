"""Build the concrete tensors: the board graph (adjacency matrix) and the empty board."""

import torch

BLACK, WHITE = 1.0, -1.0


def adjacency(n: int, device: torch.device) -> torch.Tensor:
    """A_{pq} = 1 if points p and q are horizontal or vertical neighbours on the n×n board: shape (n², n²)."""
    a = torch.zeros(n * n, n * n, device=device)
    for r in range(n):
        for c in range(n):
            p = r * n + c
            if c + 1 < n:
                a[p, p + 1] = a[p + 1, p] = 1
            if r + 1 < n:
                a[p, p + n] = a[p + n, p] = 1
    return a


def empty_board(n: int, device: torch.device) -> torch.Tensor:
    """X_p ∈ {+1 black, −1 white, 0 empty} for the n² points, flattened row by row: shape (n²,)."""
    return torch.zeros(n * n, device=device)
