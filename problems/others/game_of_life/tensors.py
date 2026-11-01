"""Build the concrete tensors: the neighbour kernel and the starting boards (classic patterns or random soups)."""

import re

import torch

# Classic patterns in run-length encoding (RLE): b = dead, o = alive, $ = next row, a number repeats the next item.
PATTERNS = {
    "glider": "bo$2bo$3o!",
    "gosper_gun": (
        "24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!"
    ),
    "r_pentomino": "b2o$2o$bo!",
    "pulsar": "2b3o3b3o2$o4bobo4bo$o4bobo4bo$o4bobo4bo$2b3o3b3o2$2b3o3b3o$o4bobo4bo$o4bobo4bo$o4bobo4bo2$2b3o3b3o!",
}


def kernel(device: torch.device) -> torch.Tensor:
    """K_{ab}: ones on the 3×3 neighbourhood with a zero in the middle (a cell is not its own neighbour): (3, 3)."""
    k = torch.ones(3, 3, device=device)
    k[1, 1] = 0
    return k


def parse_rle(rle: str) -> torch.Tensor:
    """Decode an RLE string into a 0/1 tensor of shape (rows, cols)."""
    rows, row = [], []
    for count, tag in re.findall(r"(\d*)([bo$!])", rle):
        n = int(count) if count else 1
        if tag in "bo":
            row += [1.0 if tag == "o" else 0.0] * n
        else:  # "$" ends a row (n − 1 extra empty rows), "!" ends the pattern
            rows.append(row)
            rows += [[] for _ in range(n - 1)] if tag == "$" else []
            row = []
    width = max(len(r) for r in rows)
    return torch.tensor([r + [0.0] * (width - len(r)) for r in rows])


def board(pattern: str, size: int, n_boards: int, density: float, seed: int, device: torch.device) -> torch.Tensor:
    """Starting boards X_{bij} of shape (B, size, size); B = n_boards random soups, or 1 board with a pattern."""
    if pattern == "random":
        gen = torch.Generator().manual_seed(seed)
        return (torch.rand(n_boards, size, size, generator=gen) < density).float().to(device)
    cells = parse_rle(PATTERNS[pattern])
    x = torch.zeros(1, size, size)
    r0 = 2 if pattern == "gosper_gun" else (size - cells.shape[0]) // 2  # the gun sits top-left, firing down-right
    c0 = 2 if pattern == "gosper_gun" else (size - cells.shape[1]) // 2
    x[0, r0 : r0 + cells.shape[0], c0 : c0 + cells.shape[1]] = cells
    return x.to(device)
