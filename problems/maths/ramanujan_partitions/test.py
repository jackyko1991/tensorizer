"""Mathematical checks, written first (TDD): what any correct partition-function code must satisfy.

Run with `pytest problems/maths/ramanujan_partitions/test.py`. Each test is a fact about p(n), not an implementation
detail: the tensor recurrence agrees with a slow big-integer product formula and with the known value
p(100) = 190569292, Ramanujan's congruences hold, the Hardy–Ramanujan formula is asymptotic to p(n), and the
Rademacher series rounds to the exact value.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _partitions_slow(n_max: int) -> list[int]:
    """Reference: the coefficients of Π_k 1/(1 − q^k), multiplying in one factor at a time with Python big ints."""
    p = [1] + [0] * n_max
    for k in range(1, n_max + 1):
        for n in range(k, n_max + 1):
            p[n] += p[n - k]  # multiply by 1/(1 − q^k) = 1 + q^k + q^{2k} + …
    return p


def test_known_values():
    p = solve.partitions(200, CPU, torch.int64)
    assert p[:11].tolist() == [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]
    assert int(p[100]) == 190569292  # MacMahon's table, used by Hardy and Ramanujan in 1918
    assert int(p[200]) == 3972999029388


def test_int64_recurrence_matches_slow_product():
    n_max = tensors.N_EXACT  # the largest n with p(n) < 2^63
    assert solve.partitions(n_max, CPU, torch.int64).tolist() == _partitions_slow(n_max)


def test_chinese_remainder_reconstruction_matches_big_integers():
    """Beyond int64: residues modulo several primes near 2^31, glued back together by the Chinese remainder theorem."""
    ref = _partitions_slow(1200)
    primes = tensors.crt_primes(1200, CPU)
    residues = solve.partitions(1200, CPU, torch.int64, primes)  # (N + 1, M)
    digits = equation.garner_digits(residues, primes)  # mixed-radix digits, (N + 1, M)
    assert all(equation.digits_to_int(digits[n], primes) == ref[n] for n in range(0, 1201, 37))
    p = equation.digits_to_float(digits, primes)
    rel = (p[1:] - torch.tensor([float(x) for x in ref[1:]], dtype=torch.float64)).abs() / p[1:]
    assert rel.max() < 1e-14


def test_naive_float_recurrence_is_unstable():
    """Why the residues: in float64 the signed pentagonal sum loses digits, and the loss grows like e^{c√n}."""
    ref = _partitions_slow(1000)
    p = solve.partitions(1000, CPU, torch.float64)
    assert abs(float(p[200]) / ref[200] - 1) < 1e-13
    assert abs(float(p[1000]) / ref[1000] - 1) > 1e-6


def test_residue_tables_match_exact_values():
    moduli = tensors.moduli(CPU)  # (M,)
    p_mod = solve.partitions(300, CPU, torch.int64, moduli)  # (N + 1, M)
    exact = solve.partitions(300, CPU, torch.int64)
    assert torch.equal(p_mod, exact[:, None] % moduli[None, :])


def test_ramanujan_congruences():
    a, b, m = tensors.congruences(CPU).T  # p(a k + b) ≡ 0 (mod m)
    p_mod = solve.partitions(5000, CPU, torch.int64, m)
    held, checked = equation.congruence_holds(p_mod, a, b)
    assert held.all()
    assert (checked > 40).all()
    # other residue classes are not special: p(5k) is often not divisible by 5
    assert (p_mod[0::5, 0] != 0).float().mean() > 0.5


def test_hardy_ramanujan_ratio_tends_to_one():
    p = solve.partitions_float(10000, CPU)
    n = torch.tensor([10, 100, 1000, 10000])
    ratio = p[n] / equation.hardy_ramanujan(n.double())
    gap = (ratio - 1).abs()
    assert (gap[1:] < gap[:-1]).all()  # closer to 1 at every decade
    assert gap[-1] < 0.015
    # the relative error decays like 1/√n: (1 − ratio)·√n settles near a constant
    # expanding the first Rademacher term: 1 − ratio ≈ (1/C + C/48)/√n with C = π√(2/3)
    scaled = (1 - ratio) * n.double().sqrt()
    assert abs(scaled[-1] - (1 / equation.C + equation.C / 48)) < 0.003


def test_rademacher_series_rounds_to_the_exact_value():
    n = torch.arange(1, 201)
    terms = equation.rademacher_terms(n, 12)  # (N, K)
    approx = terms.sum(dim=1)
    exact = solve.partitions(200, CPU, torch.int64)[1:]
    assert torch.equal(approx.round().long(), exact)
    assert (approx - exact.double()).abs().max() < 0.2
    # the first term alone is already the Hardy–Ramanujan leading term with the right correction
    first = terms[:, 0]
    assert abs(first[99] / 190569292 - 1) < 1e-5


def test_kloosterman_sums_and_dedekind_sums():
    n = torch.arange(0, 12)
    a = equation.kloosterman_a(n, 4)  # (N, K) for k = 1 … 4
    assert torch.allclose(a[:, 0], torch.ones(12, dtype=torch.float64))  # A_1(n) = 1
    assert torch.allclose(a[:, 1], (-1.0) ** n.double())  # A_2(n) = (−1)^n
    # Dedekind sum reciprocity: s(h, k) + s(k, h) = −1/4 + (h/k + k/h + 1/(hk))/12 for coprime h, k
    for h, k in [(3, 7), (5, 11), (2, 9)]:
        lhs = float(
            equation.dedekind_sum(torch.tensor(h), torch.tensor(k))
            + equation.dedekind_sum(torch.tensor(k), torch.tensor(h))
        )
        assert math.isclose(lhs, -0.25 + (h / k + k / h + 1 / (h * k)) / 12, abs_tol=1e-12)
