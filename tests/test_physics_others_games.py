"""Checks for the Monte Carlo π and AlphaGo-mini problems."""

import math

import torch

from problems.others.alphago_mini import equation as go
from problems.others.alphago_mini import solve as go_solve
from problems.others.alphago_mini import tensors as go_t
from problems.others.monte_carlo_pi import equation as mc
from problems.others.monte_carlo_pi import solve as mc_solve
from problems.others.monte_carlo_pi import tensors as mc_t

CPU = torch.device("cpu")


# --- Monte Carlo π ------------------------------------------------------------------------------


def test_darts_converge_with_the_binomial_error():
    est = mc_solve.darts_estimate(mc_t.darts(64, 20000, 0, CPU))
    sd = mc_solve.spread(est)
    assert abs(est[:, -1].mean().item() - math.pi) < 4 * sd[-1].item() / 8  # mean of 64 runs: σ/8
    ratio = sd[-1] / mc.standard_error(torch.tensor(20000.0, dtype=torch.float64))
    assert 0.7 < ratio.item() < 1.3


def test_running_estimate_is_a_prefix_mean():
    h = torch.tensor([[1.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(mc.running_estimate(h, 4.0), torch.tensor([[4.0, 2.0, 8 / 3, 3.0]], dtype=torch.float64))


def test_buffon_needle_estimates_pi():
    y, phi = mc_t.needles(16, 50000, 0, CPU)
    assert abs(mc_solve.needle_estimate(y, phi, 0.8).mean().item() - math.pi) < 0.02


# --- Go rules and search ------------------------------------------------------------------------

N = 5
ADJ = go_t.adjacency(N, CPU)


def board(rows: list[str]) -> torch.Tensor:
    return torch.tensor([{"X": 1.0, "O": -1.0, ".": 0.0}[ch] for row in rows for ch in row])


def test_adjacency_is_the_grid_graph():
    assert torch.equal(ADJ, ADJ.T)
    assert ADJ.sum().item() == 2 * 2 * N * (N - 1)  # 2·n(n−1) edges, each counted twice
    assert ADJ[0].sum() == 2 and ADJ[N + 1].sum() == 4


def test_capture_removes_a_surrounded_group():
    b = board(["XX...", "OO...", ".....", ".....", "....."])  # the black pair has one liberty left, at (0,2)
    after = go.play(b[None], torch.tensor([-1.0]), torch.tensor([2]), ADJ)[0]  # white fills it
    assert torch.equal(after, board(["..O..", "OO...", ".....", ".....", "....."]))


def test_suicide_is_illegal_and_ko_is_blocked():
    b = board([".O...", "O....", ".....", ".....", "....."])
    legal, _ = go_solve.legal_moves(b, 1.0, None, ADJ)
    assert not legal[0]  # black at the corner would have no liberty
    assert legal[N * N]  # passing is always legal
    before = board([".XO..", "XO.O.", ".XO..", ".....", "....."])
    after = go.play(before[None], torch.tensor([1.0]), torch.tensor([1 * N + 2]), ADJ)[0]  # black captures at (1,2)
    assert after[1 * N + 1] == 0
    legal, _ = go_solve.legal_moves(after, -1.0, before, ADJ)
    assert not legal[1 * N + 1]  # white may not retake the ko immediately


def test_area_score_counts_stones_and_surrounded_points():
    b = board(["..X..", "..X..", "..XO.", "..XO.", "..XO."])
    # black: 5 stones + the 10 points left of them; white: 3 stones; the right-hand empty region touches both
    assert go.score(b[None], ADJ, 0.0).item() == (5 + 10) - 3


def test_search_prefers_capturing_a_group_in_atari():
    b = board([".X...", "XOX..", ".....", ".....", "....."])  # white at (1,1) in atari: black (2,1) captures
    root = go_solve.Node(b, 1.0, None)
    gen = torch.Generator().manual_seed(0)
    counts = go_solve.search(root, ADJ, 96, 1.5, 32, N * N, 2.5, gen)
    assert int(counts.argmax()) == 2 * N + 1
