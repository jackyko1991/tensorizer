"""Checks for the Game of Life problem: known still lifes, oscillators, the glider and the Gosper gun."""

import torch

from problems.others.game_of_life import equation, solve, tensors

CPU = torch.device("cpu")
K = tensors.kernel(CPU)


def place(rle: str, size: int = 12, at: tuple[int, int] = (4, 4)) -> torch.Tensor:
    cells = tensors.parse_rle(rle)
    x = torch.zeros(1, size, size)
    x[0, at[0] : at[0] + cells.shape[0], at[1] : at[1] + cells.shape[1]] = cells
    return x


def test_neighbours_match_conv2d():
    x = (torch.rand(3, 20, 20, generator=torch.Generator().manual_seed(1)) < 0.4).float()
    conv = torch.nn.functional.conv2d(x[:, None], K[None, None], padding=1)[:, 0]
    assert torch.equal(equation.neighbours(x, K), conv)


def test_block_is_still_and_blinker_has_period_two():
    block = place("2o$2o!")
    assert torch.equal(equation.step(block, K), block)
    blinker = place("3o!")
    once = equation.step(blinker, K)
    assert not torch.equal(once, blinker)
    assert torch.equal(equation.step(once, K), blinker)


def test_pulsar_has_period_three():
    x = tensors.board("pulsar", 25, 1, 0.0, 0, CPU)
    history = solve.evolve(x, K, 3)
    assert torch.equal(history[3], history[0]) and not torch.equal(history[1], history[0])


def test_glider_moves_one_cell_diagonally_every_four_generations():
    x = place(tensors.PATTERNS["glider"], size=20, at=(2, 2))
    history = solve.evolve(x, K, 4)
    assert torch.equal(history[4, 0], torch.roll(history[0, 0], shifts=(1, 1), dims=(0, 1)))


def test_gosper_gun_adds_one_glider_per_period():
    x = tensors.board("gosper_gun", 100, 1, 0.0, 0, CPU)
    pop = solve.population(solve.evolve(x, K, 120))[:, 0]
    assert pop[0] == 36
    assert [int(pop[30 * n] - pop[0]) for n in range(1, 5)] == [5, 10, 15, 20]


def test_random_soups_are_reproducible_and_batched():
    a = tensors.board("random", 16, 4, 0.3, 0, CPU)
    assert a.shape == (4, 16, 16) and torch.equal(a, tensors.board("random", 16, 4, 0.3, 0, CPU))
    assert torch.equal(equation.step(a, K)[2:3], equation.step(a[2:3], K))
