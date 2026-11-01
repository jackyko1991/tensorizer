r"""Rules → tensor translation.

Conway's Game of Life is a cellular automaton, not physics, but it is the purest example of a local update rule
written as a tensor operation. Each cell X_{ij} ∈ {0, 1} counts its eight live neighbours,

    N_{ij} = \sum_{a,b} K_{ab}\,X_{i+a-1,\,j+b-1},       K = [[1, 1, 1], [1, 0, 1], [1, 1, 1]],

which is a convolution. Writing it as one einsum: `unfold` copies the 3×3 patch around every cell into a column,
P_{bkp} (k = patch position, p = cell), and the count is N_{bp} = K_k P_{bkp}. Then all cells update at once (B3/S23):

    X'_{ij} = [N_{ij} = 3] + X_{ij}\,[N_{ij} = 2]

— a dead cell with exactly 3 neighbours is born, a live cell with 2 or 3 survives, every other cell dies.
Cells outside the board are dead.
"""

import torch
import torch.nn.functional as F


def neighbours(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""N_{bp} = K_k P_{bkp}   (k summed over the 9 patch positions).   x: (B, H, W) → (B, H, W)"""
    b, h, w = x.shape
    patches = F.unfold(x[:, None], kernel_size=3, padding=1)  # (B, 9, H·W)
    return torch.einsum("k,bkp->bp", k.flatten(), patches).reshape(b, h, w)


def step(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""X' = [N = 3] + X·[N = 2]: one generation for every board.   (B, H, W) → (B, H, W)"""
    n = neighbours(x, k)
    return ((n == 3) | ((x > 0) & (n == 2))).float()
