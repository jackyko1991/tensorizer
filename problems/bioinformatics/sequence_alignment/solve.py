"""Fill the DP tables diagonal by diagonal for the whole batch, read off the scores and trace back the alignment."""

import torch

from problems._common.progress import steps

from .equation import wavefront_step


def init_tables(S: torch.Tensor, gap_open: float, gap_extend: float, local: bool):
    """H, E, F of shape (B, n+1, m+1) with the empty-prefix row and column filled in."""
    B, n1, m1 = S.shape
    H = torch.zeros(B, n1, m1, device=S.device)
    E = torch.full_like(H, float("-inf"))
    F = torch.full_like(H, float("-inf"))
    if not local:  # Needleman–Wunsch: a prefix aligned to nothing is one gap of that length
        gap = lambda k: -(gap_open + (k - 1) * gap_extend)  # noqa: E731
        H[:, 0, 1:] = gap(torch.arange(1, m1, device=S.device))
        H[:, 1:, 0] = gap(torch.arange(1, n1, device=S.device))
        E[:, 0, 1:] = H[:, 0, 1:]
        F[:, 1:, 0] = H[:, 1:, 0]
    return H, E, F


def fill(S: torch.Tensor, gap_open: float, gap_extend: float, local: bool):
    """Run the wavefront over the anti-diagonals d = 2 … n+m. S: (B, n+1, m+1) → H, E, F, each (B, n+1, m+1)."""
    H, E, F = init_tables(S, gap_open, gap_extend, local)
    n, m = S.shape[1] - 1, S.shape[2] - 1
    for d in steps(n + m - 1, "anti-diagonals"):
        d += 2
        i = torch.arange(max(1, d - m), min(n, d - 1) + 1, device=S.device)  # (K,) rows on this diagonal
        j = d - i  # (K,) matching columns
        H[:, i, j], E[:, i, j], F[:, i, j] = wavefront_step(H, E, F, S, i, j, gap_open, gap_extend, local)
    return H, E, F


def best_cells(H: torch.Tensor, len_a: torch.Tensor, len_c: torch.Tensor, local: bool):
    """Score and end cell of every pair: (B,), (B, 2).

    Global: the bottom-right cell (n_b, m_b) of each pair's own lengths. Local: the best cell of that pair's table.
    """
    B = H.shape[0]
    if not local:
        ends = torch.stack([len_a, len_c], dim=1)
        return H[torch.arange(B), len_a, len_c], ends
    rows = torch.arange(H.shape[1], device=H.device)
    cols = torch.arange(H.shape[2], device=H.device)
    inside = (rows[None, :, None] <= len_a[:, None, None]) & (cols[None, None, :] <= len_c[:, None, None])
    flat = torch.where(inside, H, float("-inf")).flatten(1)  # padding cells are ignored
    score, k = flat.max(dim=1)
    return score, torch.stack([k // H.shape[2], k % H.shape[2]], dim=1)


def traceback(H, E, F, S, a: str, c: str, end: tuple[int, int], gap_open, gap_extend, local: bool):
    """Walk back from the end cell of one pair, re-deriving each choice from the tables.

    Returns the path [(i, j), …] from start to end and the two aligned strings with '-' for gaps.
    """
    H, E, F, S = (t.cpu().tolist() for t in (H, E, F, S))  # one pair: (n+1, m+1) nested lists
    i, j = end
    state, path, top, bottom = "H", [(i, j)], [], []
    while i > 0 or j > 0:
        if local and state == "H" and H[i][j] <= 0:
            break
        if state == "H" and i > 0 and j > 0 and H[i][j] == H[i - 1][j - 1] + S[i][j]:
            top.append(a[i - 1])
            bottom.append(c[j - 1])
            i, j = i - 1, j - 1
        elif state == "H":
            state = "E" if j > 0 and (i == 0 or H[i][j] == E[i][j]) else "F"
            continue
        elif state == "E":  # gap in a: c_j is aligned to '-'
            state = "H" if i == 0 or E[i][j] == H[i][j - 1] - gap_open else "E"
            top.append("-")
            bottom.append(c[j - 1])
            j -= 1
        else:  # gap in c: a_i is aligned to '-'
            state = "H" if j == 0 or F[i][j] == H[i - 1][j] - gap_open else "F"
            top.append(a[i - 1])
            bottom.append("-")
            i -= 1
        path.append((i, j))
    return path[::-1], "".join(top[::-1]), "".join(bottom[::-1])
