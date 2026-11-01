r"""Alignment rules → tensor translation.

Two DNA sequences a (length n) and c (length m) are aligned by dynamic programming. With one-hot residues
a_{bi\alpha} (pair b, position i, letter \alpha) and the substitution matrix W_{\alpha\beta}, every match score is

    S_{bij} = a_{bi\alpha}\,W_{\alpha\beta}\,c_{bj\beta}        (\alpha, \beta summed over A, C, G, T)

A gap of length k costs g_o + (k − 1) g_e (affine; g_o = g_e is a linear gap). Gotoh's recurrences keep three tables:
H (best score ending at cell i, j), E (ending with a gap in a) and F (ending with a gap in c):

    E_{bij} = max(H_{b,i,j-1} − g_o,  E_{b,i,j-1} − g_e)
    F_{bij} = max(H_{b,i-1,j} − g_o,  F_{b,i-1,j} − g_e)
    H_{bij} = max(H_{b,i-1,j-1} + S_{bij},  E_{bij},  F_{bij}  [, 0 for Smith–Waterman])

Cell (i, j) only needs cells with a smaller i + j. So every cell on one anti-diagonal i + j = d can be computed at
the same time from the two previous diagonals: the "wavefront". One diagonal is one tensor update, for every cell
on it and every pair b in the batch.
"""

import torch


def substitution_scores(a: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    r"""S_{bij} = a_{bi\alpha} W_{\alpha\beta} c_{bj\beta}   (\alpha, \beta summed): (B, n, 4), (B, m, 4) → (B, n, m)"""
    return torch.einsum("bia,ay,bjy->bij", a, w, c)


def wavefront_step(H, E, F, S, i, j, gap_open: float, gap_extend: float, local: bool):
    r"""New H, E, F on one anti-diagonal: cells (i_k, j_k) with i_k + j_k = d, for every pair at once.

    H, E, F, S: (B, n+1, m+1) tables (row and column 0 are the empty prefix); i, j: (K,) cell indices.
    Returns three (B, K) tensors.
    """
    e = torch.maximum(H[:, i, j - 1] - gap_open, E[:, i, j - 1] - gap_extend)  # gap in a: step from the left
    f = torch.maximum(H[:, i - 1, j] - gap_open, F[:, i - 1, j] - gap_extend)  # gap in c: step from above
    h = torch.maximum(torch.maximum(H[:, i - 1, j - 1] + S[:, i, j], e), f)  # match/mismatch: step diagonally
    if local:
        h = h.clamp(min=0.0)  # Smith–Waterman: a local alignment may start afresh anywhere
    return h, e, f
