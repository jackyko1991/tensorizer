"""Mathematical checks, written first (TDD): what any correct batched Turing-machine simulator must satisfy.

Run with `pytest problems/others/turing_halting/test.py`. Each test is a known fact about Turing machines, not an
implementation detail: the batched tensor step agrees with a plain Python simulator, the exhaustive search over all
2-state machines finds the busy beaver values S(2) = 6 and Σ(2) = 4, the published champions halt after exactly
21 and 107 steps, a machine that only moves right never halts, mirror images halt together, and the fraction of
machines that halted can only grow with the step budget.
"""

import random

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def reference(table: list, n_steps: int) -> list[tuple[dict, int, int]]:
    """A tiny pure-Python Turing machine: tape as a dict, one rule look-up per step. Returns the state history."""
    n = len(table)
    tape, head, state, history = {}, 0, 0, []
    for _ in range(n_steps):
        if state < n:  # a halted machine (state n = H) stays frozen
            write, move, nxt = table[state][tape.get(head, 0)]
            tape[head] = write
            head, state = head + move, nxt
        history.append((dict(tape), head, state))
    return history


def test_batched_step_matches_python_reference():
    rng = random.Random(3)
    n, b, n_steps = 3, 64, 40
    rows = [
        [[[rng.randint(0, 1), rng.choice((-1, 1)), rng.randint(0, n)] for _ in range(2)] for _ in range(n)]
        for _ in range(b)
    ]
    table = torch.tensor(rows)  # (B, n, 2, 3)
    length = 2 * n_steps + 1
    tape, head, state = tensors.blank(b, length, CPU)
    histories = [reference(r, n_steps) for r in rows]
    for t in range(n_steps):
        active = state < n
        tape, head, state = equation.step(tape, head, state, table, active)
        for i in range(b):
            ref_tape, ref_head, ref_state = histories[i][t]
            assert int(state[i]) == ref_state and int(head[i]) - length // 2 == ref_head
            cells = {x - length // 2: int(v) for x, v in enumerate(tape[i].tolist()) if v}
            assert cells == {x: v for x, v in ref_tape.items() if v}


def test_all_two_state_machines_give_busy_beaver_values():
    table = tensors.all_machines(2, normalise=False, device=CPU)
    assert table.shape == (12**4, 2, 2, 3)  # (2 symbols × 2 moves × 3 next states)^(2 states × 2 symbols)
    res = solve.run(table, budget=30)
    halted = res["halt_time"] >= 0
    assert int(res["halt_time"].max()) == 6  # Radó: S(2) = 6 steps
    assert int(res["ones"][halted].max()) == 4  # Σ(2) = 4 ones


def test_halted_fraction_rises_then_plateaus_below_one():
    res = solve.run(tensors.all_machines(2, normalise=False, device=CPU), budget=40)
    f = solve.halted_fraction(res["halt_time"], 40)  # (41,)
    assert f[0] == 0 and torch.all(f.diff() >= 0)  # a halted machine stays halted: non-decreasing in the budget
    assert torch.all(f[6:] == f[6]) and f[-1] < 1  # nothing halts after S(2) = 6: the rest never halts


def test_mirror_image_machines_halt_at_the_same_step():
    table = tensors.all_machines(2, normalise=False, device=CPU)
    mirror = table.clone()
    mirror[..., 1] *= -1  # swap left and right everywhere
    a, b = solve.run(table, budget=20), solve.run(mirror, budget=20)
    assert torch.equal(a["halt_time"], b["halt_time"]) and torch.equal(a["ones"], b["ones"])


@pytest.mark.parametrize(
    ("name", "steps", "ones"),
    [("BB(2)", 6, 4), ("BB(3) steps", 21, 5), ("BB(3) ones", 14, 6), ("BB(4)", 107, 13)],
)
def test_known_champions(name, steps, ones):
    table = tensors.parse(tensors.CHAMPIONS[name])[None]  # (1, n, 2, 3)
    res = solve.run(table, budget=300)
    assert int(res["halt_time"][0]) == steps and int(res["ones"][0]) == ones


def test_right_mover_never_halts():
    table = tensors.parse("1RA1RA_1RH1RH")[None]  # state A writes 1 and moves right forever; B is never reached
    res = solve.run(table, budget=500)
    assert int(res["halt_time"][0]) == -1 and int(res["ones"][0]) == 500 and not res["off_tape"][0]


def test_running_off_the_tape_counts_as_undecided():
    table = tensors.parse("1RA1RA_1RH1RH")[None]
    res = solve.run(table, budget=50, length=21)  # a tape too short for the budget
    assert bool(res["off_tape"][0]) and int(res["halt_time"][0]) == -1


@pytest.mark.slow
def test_three_state_enumeration_finds_s21_and_sigma6():
    table = tensors.all_machines(3, normalise=True, device=CPU)
    assert table.shape[0] == 13**5 and torch.all(table[:, 0, 0] == torch.tensor([1, 1, 1]))  # A0 = 1RB fixed
    res = solve.run(table, budget=60)
    halted = res["halt_time"] >= 0
    assert int(res["halt_time"].max()) == 21 and int(res["ones"][halted].max()) == 6  # S(3) = 21, Σ(3) = 6
    f = solve.halted_fraction(res["halt_time"], 60)
    assert torch.all(f[21:] == f[21]) and f[-1] < 1
