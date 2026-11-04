"""Probability checks, written first (TDD): what any correct Monty Hall simulation must satisfy.

Run with `pytest problems/others/monty_hall/test.py`. Each test is a law of probability, not an implementation
detail: with a host who knows where the car is and opens N − 2 goat doors, staying wins with probability 1/N and
switching with (N − 1)/N (2/3 for the classic three doors); the answer does not depend on the first pick; an
ignorant host who happens not to reveal the car makes both choices 1/2; and a slow game-by-game reference agrees.
"""

import math
import random

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def sigma(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)  # binomial standard error of a win rate measured over n games


def test_classic_three_doors_switching_wins_two_thirds():
    car, pick, kept = tensors.games(torch.tensor([3]), 400_000, "knowing", 0, CPU)
    stay, switch, valid = equation.outcomes(car, pick, kept, "knowing")
    n = 400_000
    assert valid.all()  # a knowing host never reveals the car
    assert abs(switch.double().mean().item() - 2 / 3) < 4 * sigma(2 / 3, n)
    assert abs(stay.double().mean().item() - 1 / 3) < 4 * sigma(1 / 3, n)
    assert torch.equal(stay ^ switch, torch.ones_like(stay))  # exactly one of the two closed doors hides the car


def test_many_doors_rates_match_one_over_n():
    doors = torch.tensor([3, 4, 10, 100, 1000])
    n = 200_000
    rates = solve.final_rates(doors, n, "knowing", 1, CPU)  # (2, B): stay, switch
    for b, d in enumerate(doors.tolist()):
        assert abs(rates[0, b].item() - 1 / d) < 4 * sigma(1 / d, n) + 1e-9  # stay: you were right from the start
        assert abs(rates[1, b].item() - (d - 1) / d) < 4 * sigma(1 / d, n) + 1e-9  # switch: (N − 1)/N


def test_win_rate_does_not_depend_on_the_first_pick():
    n = 300_000
    for d in (3, 5):
        car, pick, kept = tensors.games(torch.tensor([d]), n, "knowing", 2, CPU)
        _, switch, _ = equation.outcomes(car, pick, kept, "knowing")
        for door in range(d):
            chosen = pick == door  # games whose first pick was this door
            m = int(chosen.sum())
            assert abs(switch[chosen].double().mean().item() - (d - 1) / d) < 4 * sigma(1 / d, m)


def test_host_opens_only_goat_doors_and_never_the_pick():
    car, pick, kept = tensors.games(torch.tensor([7]), 50_000, "knowing", 3, CPU)
    opened = equation.opened_doors(pick, kept, 7)  # (B, G, N) bool
    assert torch.equal(opened.sum(-1), torch.full_like(pick, 5))  # exactly N − 2 doors opened
    assert not opened.gather(-1, pick[..., None]).any()  # the contestant's door stays shut
    assert not opened.gather(-1, car[..., None]).any()  # a knowing host never shows the car


def test_ignorant_host_gives_one_half():
    n = 400_000
    car, pick, kept = tensors.games(torch.tensor([3, 10]), n, "ignorant", 4, CPU)
    stay, switch, valid = equation.outcomes(car, pick, kept, "ignorant")
    for b, d in enumerate((3, 10)):
        m = int(valid[b].sum())
        assert abs(m / n - 2 / d) < 4 * sigma(2 / d, n)  # the car survives the random opening with probability 2/N
        assert abs(switch[b][valid[b]].double().mean().item() - 0.5) < 4 * sigma(0.5, m)  # then it is a coin flip
        assert abs(stay[b][valid[b]].double().mean().item() - 0.5) < 4 * sigma(0.5, m)


def test_agrees_with_a_slow_game_by_game_reference():
    rng = random.Random(5)
    d, n = 4, 20_000
    wins = 0
    for _ in range(n):  # the literal story, one game at a time
        car, pick = rng.randrange(d), rng.randrange(d)
        goats = [k for k in range(d) if k != pick and k != car]
        opened = rng.sample(goats, d - 2)  # all goats but one if you picked the car, else every goat
        other = next(k for k in range(d) if k != pick and k not in opened)
        wins += other == car
    slow = wins / n
    fast = solve.final_rates(torch.tensor([d]), n, "knowing", 6, CPU)[1, 0].item()
    assert abs(slow - fast) < 4 * math.sqrt(2) * sigma(3 / 4, n)  # two independent estimates of the same rate


def test_running_rate_is_a_prefix_mean_with_binomial_band():
    wins = torch.tensor([[True, False, True, True]])
    assert torch.allclose(equation.running_rate(wins), torch.tensor([[1.0, 0.5, 2 / 3, 0.75]], dtype=torch.float64))
    band = equation.binomial_band(torch.tensor(0.5, dtype=torch.float64), torch.tensor([100.0], dtype=torch.float64))
    assert torch.allclose(band, torch.tensor([0.1], dtype=torch.float64))  # 2σ = 2·√(¼/100)
