"""Mathematical checks, written first (TDD): what any correct 3-SAT verifier and exhaustive search must satisfy.

Run with `pytest problems/maths/p_vs_np_sat/test.py`. Each test is a fact about Boolean formulas, not an
implementation detail: the cheap verifier and the exhaustive search agree, one 3-clause removes exactly 1/8 of all
assignments, the expected number of solutions is 2^n (7/8)^M, adding clauses can only destroy solutions, and the
satisfiable fraction collapses near M/n ≈ 4.27.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_verifier_agrees_with_brute_force():
    n, m, b = 8, 36, 64  # ratio 4.5: a mix of satisfiable and unsatisfiable formulas
    gen = torch.Generator().manual_seed(1)
    var, neg = tensors.random_3sat(n, m, b, gen, CPU)
    x_all = tensors.all_assignments(n, CPU)  # (2^n, n)
    # slow reference: run the verifier on every (formula, assignment) pair
    ok = equation.verify(x_all[None].expand(b, -1, -1), var[:, None], neg[:, None])  # (B, 2^n)
    table = solve.solution_table(var, neg, n)  # packed exhaustive search, (B, W)
    assert torch.equal(equation.unpack(table, n), ok)
    sat = ok.any(dim=1)
    assert 0 < int(sat.sum()) < b  # both outcomes occur
    # the certificate found by the search passes the polynomial-time verifier
    first = solve.first_solution(table, n)  # (B,)
    x = x_all[first.clamp(max=2**n - 1)]
    assert torch.equal(equation.verify(x, var, neg), sat)


def test_one_clause_removes_one_eighth_of_assignments():
    n = 10
    gen = torch.Generator().manual_seed(2)
    var, neg = tensors.random_3sat(n, 1, 32, gen, CPU)
    count = equation.popcount(solve.solution_table(var, neg, n))
    assert torch.all(count == 2**n * 7 // 8)  # exactly one of the 8 patterns of its 3 variables is excluded


def test_planted_formulas_are_satisfiable():
    n, m, b = 12, 72, 16  # ratio 6, far above the threshold, yet a planted solution survives
    gen = torch.Generator().manual_seed(3)
    x_star = torch.rand(b, n, generator=gen) < 0.5
    var, neg = tensors.planted_3sat(x_star, m, gen)
    assert torch.all(equation.verify(x_star, var, neg))
    assert torch.all(equation.popcount(solve.solution_table(var, neg, n)) >= 1)


def test_expected_number_of_solutions_is_first_moment():
    n, m, b = 10, 20, 400
    gen = torch.Generator().manual_seed(4)
    var, neg = tensors.random_3sat(n, m, b, gen, CPU)
    count = equation.popcount(solve.solution_table(var, neg, n)).double()
    expected = 2**n * (7 / 8) ** m  # linearity of expectation: each clause keeps an assignment with prob. 7/8
    assert abs(count.mean().item() - expected) < 4 * count.std().item() / b**0.5


def test_satisfiable_fraction_is_monotone_and_collapses_near_4_27():
    n, b = 14, 300
    gen = torch.Generator().manual_seed(5)
    var, neg = tensors.random_3sat(n, 7 * n, b, gen, CPU)
    res = solve.sweep(var, neg, n)
    p_sat = res["sat"].double().mean(dim=0)  # (M_max + 1,) nested formulas: prefixes of one clause list
    assert torch.all(p_sat.diff() <= 0)  # adding a clause can only remove solutions
    assert p_sat[2 * n] > 0.95 and p_sat[7 * n] < 0.05
    alpha_50 = solve.crossing(p_sat, n)
    assert 3.9 < alpha_50 < 5.2  # finite-size crossing near the conjectured threshold 4.267


@pytest.mark.slow
def test_transition_sharpens_with_n():
    widths = []
    for n in (10, 16):
        gen = torch.Generator().manual_seed(6)
        var, neg = tensors.random_3sat(n, 7 * n, 300, gen, CPU)
        p_sat = solve.sweep(var, neg, n)["sat"].double().mean(dim=0)
        widths.append(solve.crossing(p_sat, n, 0.1) - solve.crossing(p_sat, n, 0.9))
    assert widths[1] < 0.8 * widths[0]  # the window shrinks: a sharp threshold in the limit (Friedgut 1999)


@pytest.mark.slow
def test_backtracking_work_peaks_in_the_transition_window():
    n, b = 14, 200
    gen = torch.Generator().manual_seed(7)
    var, neg = tensors.random_3sat(n, 8 * n, b, gen, CPU)
    alphas = torch.arange(1.5, 8.01, 0.5)
    res = solve.sweep(var, neg, n, record=[round(a.item() * n) for a in alphas])
    work = res["nodes"].double().median(dim=0).values  # (len(alphas),)
    peak = alphas[work.argmax()].item()
    p_sat = res["sat"].double().mean(dim=0)[round(peak * n)]
    assert 3.8 <= peak <= 6.0 and 0.02 < p_sat < 0.98  # hardest where satisfiable and unsatisfiable formulas mix
    assert work.max() > 5 * work[0] and work.max() > 1.3 * work[-1]  # easy–hard–easy
