"""Physics checks, written first (TDD): what any correct Go engine must satisfy.

Run with `pytest problems/ai/alphago_mini/test.py`. Each test is a rule or invariant of the game, not an
implementation detail: the board is the grid graph, surrounded groups are captured, suicide and immediate ko
retakes are illegal, the area score is antisymmetric under swapping colours and invariant under the board's
symmetries, and the tree search finds a capture.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
N = 5
ADJ = tensors.adjacency(N, CPU)


def board(rows: list[str]) -> torch.Tensor:
    return torch.tensor([{"X": 1.0, "O": -1.0, ".": 0.0}[ch] for row in rows for ch in row])


def test_adjacency_is_the_grid_graph():
    assert torch.equal(ADJ, ADJ.T)  # undirected: neighbourhood is symmetric
    assert ADJ.sum().item() == 2 * 2 * N * (N - 1)  # 2·n(n−1) edges, each counted twice
    assert ADJ[0].sum() == 2 and ADJ[N + 1].sum() == 4


def test_capture_removes_a_surrounded_group():
    b = board(["XX...", "OO...", ".....", ".....", "....."])  # the black pair has one liberty left, at (0,2)
    after = equation.play(b[None], torch.tensor([-1.0]), torch.tensor([2]), ADJ)[0]  # white fills it
    assert torch.equal(after, board(["..O..", "OO...", ".....", ".....", "....."]))  # a group with no liberty dies


def test_suicide_is_illegal_and_ko_is_blocked():
    b = board([".O...", "O....", ".....", ".....", "....."])
    legal, _ = solve.legal_moves(b, 1.0, None, ADJ)
    assert not legal[0]  # black at the corner would have no liberty
    assert legal[N * N]  # passing is always legal
    before = board([".XO..", "XO.O.", ".XO..", ".....", "....."])
    after = equation.play(before[None], torch.tensor([1.0]), torch.tensor([1 * N + 2]), ADJ)[0]  # black takes at (1,2)
    assert after[1 * N + 1] == 0
    legal, _ = solve.legal_moves(after, -1.0, before, ADJ)
    assert not legal[1 * N + 1]  # white may not retake the ko immediately


def test_area_score_counts_stones_and_surrounded_points():
    b = board(["..X..", "..X..", "..XO.", "..XO.", "..XO."])
    # black: 5 stones + the 10 points left of them; white: 3 stones; the right-hand empty region touches both
    assert equation.score(b[None], ADJ, 0.0).item() == (5 + 10) - 3


def test_score_is_colour_antisymmetric_and_symmetry_invariant():
    gen = torch.Generator().manual_seed(0)
    boards = torch.randint(-1, 2, (16, N * N), generator=gen).float()  # (B, P) random positions
    s = equation.score(boards, ADJ, 0.0)
    assert torch.equal(equation.score(-boards, ADJ, 0.0), -s)  # swapping black and white negates the score
    grid = boards.view(-1, N, N)
    for g in (torch.rot90(grid, 1, (1, 2)), torch.flip(grid, (2,)), grid.transpose(1, 2)):
        assert torch.equal(equation.score(g.reshape(-1, N * N), ADJ, 0.0), s)  # dihedral symmetry of the board
    assert (s.abs() <= N * N).all()  # bound: a side owns at most every point
    assert equation.score(boards, ADJ, 7.5).sub(s).eq(-7.5).all()  # komi is a constant offset


def test_search_prefers_capturing_a_group_in_atari():
    b = board([".X...", "XOX..", ".....", ".....", "....."])  # white at (1,1) in atari: black (2,1) captures
    root = solve.Node(b, 1.0, None)
    gen = torch.Generator().manual_seed(0)
    counts = solve.search(root, ADJ, 96, 1.5, 32, N * N, 2.5, gen)
    assert int(counts.argmax()) == 2 * N + 1
