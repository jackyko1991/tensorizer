"""Mathematical checks, written first (TDD): what any correct cellular automaton code must satisfy.

Run with `pytest problems/others/cellular_automata/test.py`. Each test is a known property of the rules, not an
implementation detail: the lookup agrees with a slow cell-by-cell reference, rule 90 grown from one cell is
Pascal's triangle mod 2, rule 184 (traffic) conserves the number of cars, rules 0 and 255 wipe the line, the
256 rules fall into 88 classes under mirror and complement symmetry, and the Lenia glider Orbium keeps its mass
while it moves.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
TABLE = tensors.rule_table(torch.arange(256))  # (256, 8)


def slow_step(row: list[int], rule: int) -> list[int]:
    """The literal definition on a ring: new cell = bit (4·left + 2·centre + right) of the rule number."""
    w = len(row)
    return [(rule >> (4 * row[(i - 1) % w] + 2 * row[i] + row[(i + 1) % w])) & 1 for i in range(w)]


def test_lookup_matches_a_slow_reference():
    x = tensors.random_rows(256, 37, 0.5, 0, CPU)
    fast = equation.eca_step(x, TABLE)
    for r in (0, 30, 54, 90, 110, 150, 184, 255, 77, 201):
        assert fast[r].tolist() == slow_step(x[r].tolist(), r)


def test_rule_90_from_one_cell_is_pascals_triangle_mod_2():
    t_max, w = 40, 101
    hist = solve.run_eca(tensors.single_cell(1, w, CPU), TABLE[90:91], t_max)[:, 0]  # (T+1, W)
    centre = w // 2
    for t in range(t_max + 1):
        expected = torch.zeros(w, dtype=torch.uint8)
        for j in range(-t, t + 1, 2):  # cell centre + j is binom(t, (t + j)/2) mod 2: Sierpinski's triangle
            expected[centre + j] = math.comb(t, (t + j) // 2) % 2
        assert torch.equal(hist[t], expected)


def test_rule_184_conserves_the_number_of_cars():
    x = tensors.random_rows(5, 200, 0.37, 1, CPU)
    hist = solve.run_eca(x, TABLE[184:185].expand(5, 8), 150)  # (T+1, 5, W)
    cars = hist.sum(-1)  # (T+1, 5)
    assert torch.equal(cars, cars[:1].expand_as(cars))  # a car moves right only into an empty cell: a conservation law


def test_trivial_rules():
    x = tensors.random_rows(256, 50, 0.5, 2, CPU)
    y = equation.eca_step(x, TABLE)
    assert not y[0].any() and y[255].all()  # rule 0 kills every cell, rule 255 fills every cell
    assert torch.equal(y[204], x[204])  # rule 204 = 11001100₂ copies the centre cell: the identity
    assert torch.equal(y[170], torch.roll(x[170], -1))  # rule 170 copies the right neighbour: a shift


def test_mirror_and_complement_symmetries():
    x = tensors.random_rows(256, 64, 0.5, 3, CPU)
    m, c = equation.mirror_rule(torch.arange(256)), equation.complement_rule(torch.arange(256))
    assert m[30] == 86 and c[30] == 135 and m[110] == 124 and c[110] == 137  # Wolfram's equivalents
    mirrored = solve.run_eca(x.flip(-1), TABLE[m], 20).flip(-1)  # mirror rule on the reversed line ...
    assert torch.equal(mirrored, solve.run_eca(x, TABLE, 20))  # ... is the reversed evolution
    complemented = 1 - solve.run_eca(1 - x, TABLE[c], 20)  # complement rule on the inverted line ...
    assert torch.equal(complemented, solve.run_eca(x, TABLE, 20))
    assert solve.equivalence_classes(torch.arange(256)).unique().numel() == 88  # 256 rules, 88 distinct behaviours


def test_all_rules_at_once_equal_one_at_a_time():
    x = tensors.random_rows(256, 40, 0.5, 4, CPU)
    batch = solve.run_eca(x, TABLE, 10)
    for r in (30, 110):
        assert torch.equal(batch[:, r], solve.run_eca(x[r : r + 1], TABLE[r : r + 1], 10)[:, 0])


def test_behaviour_classes_of_famous_rules():
    kind = solve.behaviour(
        tensors.random_rows(256, 151, 0.5, 5, CPU), TABLE, 600
    )  # (256,) 0 uniform 1 periodic 2 other
    assert kind[0] == 0 and kind[32] == 0 and kind[255] == 0  # Wolfram class 1: everything dies or fills
    assert kind[4] == 1 and kind[184] == 1 and kind[170] == 1  # class 2: still, periodic or travelling patterns
    assert kind[30] == 2 and kind[45] == 2 and kind[90] == 2  # class 3: no period on this ring within 600 steps


def test_lenia_kernel_and_growth():
    k = tensors.lenia_kernel(64, 13.0, CPU)
    assert abs(k.sum().item() - 1) < 1e-12 and k[0, 0] == 0  # a normalised ring: U is a weighted average
    u = torch.tensor([0.15, 0.0, 1.0], dtype=torch.float64)
    g = equation.growth(u, 0.15, 0.015)
    assert abs(g[0].item() - 1) < 1e-12 and abs(g[1].item() + 1) < 1e-12  # grow at μ, decay far from it


def test_orbium_glider_keeps_its_mass_and_moves():
    a = tensors.orbium(64, CPU)
    k = tensors.lenia_kernel(64, 13.0, CPU)
    frames, mass = solve.run_lenia(a, k, 600)  # (T+1, S, S), (T+1,)
    settled = mass[200:]
    assert (settled.max() - settled.min()) / settled.mean() < 0.03  # a soliton: mass constant within 3 %
    shift, overlap = solve.best_shift(frames[300], frames[600])
    assert overlap > 0.95  # the same shape, ...
    assert math.hypot(*shift) > 5  # ... moved by several cells: a glider
