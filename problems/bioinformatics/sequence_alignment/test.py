"""Physics checks, written first (TDD): what any correct sequence-alignment code must satisfy.

Run with `pytest problems/bioinformatics/sequence_alignment/test.py`. Each test is an exact invariant: the wavefront
DP equals Gotoh's recursion, textbook examples give their known scores, tracebacks reproduce the DP score, and local
scores are never negative.
"""

import math
import random

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def reference_score(a: str, c: str, match, mismatch, go, ge, local: bool) -> float:
    """Gotoh's algorithm cell by cell in plain Python (no tensors)."""
    n, m, NEG = len(a), len(c), -math.inf
    H = [[0.0] * (m + 1) for _ in range(n + 1)]
    E = [[NEG] * (m + 1) for _ in range(n + 1)]
    F = [[NEG] * (m + 1) for _ in range(n + 1)]
    if not local:
        for j in range(1, m + 1):
            H[0][j] = E[0][j] = -(go + (j - 1) * ge)
        for i in range(1, n + 1):
            H[i][0] = F[i][0] = -(go + (i - 1) * ge)
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E[i][j] = max(H[i][j - 1] - go, E[i][j - 1] - ge)
            F[i][j] = max(H[i - 1][j] - go, F[i - 1][j] - ge)
            s = match if a[i - 1] == c[j - 1] else mismatch
            H[i][j] = max(H[i - 1][j - 1] + s, E[i][j], F[i][j], 0.0 if local else NEG)
            best = max(best, H[i][j])
    return best if local else H[n][m]


def align(pairs: list[tuple[str, str]], match, mismatch, go, ge, local: bool):
    """Batched wavefront alignment of a list of string pairs: scores (B,), ends (B, 2) and the tables."""
    codes_a, len_a = tensors.pad([tensors.encode(a) for a, _ in pairs], CPU)
    codes_c, len_c = tensors.pad([tensors.encode(c) for _, c in pairs], CPU)
    W = tensors.substitution_matrix(match, mismatch, CPU)
    S = torch.zeros(len(pairs), codes_a.shape[1] + 1, codes_c.shape[1] + 1)
    S[:, 1:, 1:] = equation.substitution_scores(tensors.one_hot(codes_a), W, tensors.one_hot(codes_c))
    H, E, F = solve.fill(S, go, ge, local)
    scores, ends = solve.best_cells(H, len_a, len_c, local)
    return scores, ends, (H, E, F, S)


def score_alignment(top: str, bottom: str, match, mismatch, go, ge) -> float:
    """Score of a given gapped alignment, column by column."""
    total, gap_top, gap_bottom = 0.0, False, False
    for x, y in zip(top, bottom):
        if x == "-":
            total -= ge if gap_top else go
            gap_top, gap_bottom = True, False
        elif y == "-":
            total -= ge if gap_bottom else go
            gap_top, gap_bottom = False, True
        else:
            total += match if x == y else mismatch
            gap_top = gap_bottom = False
    return total


def random_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [
        ("".join(rng.choice("ACGT") for _ in range(rng.randint(1, 25))),
         "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 25))))
        for _ in range(n)
    ]  # fmt: skip


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("scheme", [(1, -1, 1, 1), (2, -1, 3, 1), (3, -2, 5, 0.5)])
def test_wavefront_matches_plain_python_dp(local, scheme):
    pairs = random_pairs(40, seed=hash(scheme) % 1000 + local)
    scores, _, _ = align(pairs, *scheme, local)
    expected = [reference_score(a, c, *scheme, local) for a, c in pairs]
    assert scores.tolist() == pytest.approx(expected)


def test_needleman_wunsch_textbook_example():
    # Wikipedia's Needleman–Wunsch example: match +1, mismatch −1, gap −1 → optimal score 0
    scores, ends, (H, E, F, S) = align([("GCATGCG", "GATTACA")], 1, -1, 1, 1, local=False)
    assert scores.item() == 0
    _, top, bottom = solve.traceback(H[0], E[0], F[0], S[0], "GCATGCG", "GATTACA", (7, 7), 1, 1, False)
    assert top.replace("-", "") == "GCATGCG" and bottom.replace("-", "") == "GATTACA"
    assert score_alignment(top, bottom, 1, -1, 1, 1) == 0


def test_smith_waterman_textbook_example():
    # Wikipedia's Smith–Waterman example: match +3, mismatch −3, linear gap 2 → score 13, GTT-AC / GTTGAC
    a, c = "TGTTACGG", "GGTTGACTA"
    scores, ends, (H, E, F, S) = align([(a, c)], 3, -3, 2, 2, local=True)
    assert scores.item() == 13
    _, top, bottom = solve.traceback(H[0], E[0], F[0], S[0], a, c, tuple(ends[0].tolist()), 2, 2, True)
    assert (top, bottom) == ("GTT-AC", "GTTGAC")


def test_traceback_reproduces_the_dp_score():
    scheme = (2, -1, 3, 1)
    pairs = random_pairs(20, seed=7)
    for local in (False, True):
        scores, ends, (H, E, F, S) = align(pairs, *scheme, local)
        for b, (a, c) in enumerate(pairs):
            _, top, bottom = solve.traceback(
                H[b], E[b], F[b], S[b], a, c, tuple(ends[b].tolist()), scheme[2], scheme[3], local
            )
            assert len(top) == len(bottom)
            assert score_alignment(top, bottom, *scheme) == pytest.approx(scores[b].item())
            if not local:
                assert top.replace("-", "") == a and bottom.replace("-", "") == c


def test_smith_waterman_is_non_negative_and_finds_the_planted_motif():
    codes_a, len_a, codes_c, len_c, spans = tensors.planted_pairs(24, 60, 0.1, 3, CPU)
    W = tensors.substitution_matrix(2, -2, CPU)
    S = torch.zeros(24, codes_a.shape[1] + 1, codes_c.shape[1] + 1)
    S[:, 1:, 1:] = equation.substitution_scores(tensors.one_hot(codes_a), W, tensors.one_hot(codes_c))
    H, E, F = solve.fill(S, 3, 1, local=True)
    assert (H >= 0).all()
    scores, ends = solve.best_cells(H, len_a, len_c, local=True)
    hits = 0
    for b in range(24):
        a, c = tensors.decode(codes_a[b]), tensors.decode(codes_c[b])
        path, _, _ = solve.traceback(H[b], E[b], F[b], S[b], a, c, tuple(ends[b].tolist()), 3, 1, True)
        (i0, j0), (i1, j1) = path[0], path[-1]
        sa, ea, sc, ec = spans[b].tolist()
        overlap_a = max(0, min(i1, ea) - max(i0, sa)) / (ea - sa)
        overlap_c = max(0, min(j1, ec) - max(j0, sc)) / (ec - sc)
        hits += overlap_a > 0.8 and overlap_c > 0.8
    assert hits >= 22
