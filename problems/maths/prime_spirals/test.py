"""Mathematical checks, written first (TDD): what any correct prime-spiral code must satisfy.

Run with `pytest problems/maths/prime_spirals/test.py`. Each test is a fact about the primes, not an implementation
detail: the tensor sieve agrees with trial division and with the known values of π(x) and of the twin-prime count,
the Ulam spiral visits every grid cell once with consecutive numbers side by side, Euler's polynomial gives 40 primes
in a row, the Goldbach convolution agrees with brute force, and the Hardy–Littlewood constants take their known values.
"""

import math

import torch

from . import equation, tensors

CPU = torch.device("cpu")


def _is_prime_slow(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def test_sieve_matches_trial_division():
    is_prime = tensors.sieve(3000, CPU)
    assert is_prime.tolist() == [_is_prime_slow(n) for n in range(3001)]


def test_prime_counts_and_twin_counts():
    is_prime = tensors.sieve(10**6, CPU)
    pi = torch.cumsum(is_prime.long(), 0)
    assert [int(pi[10**k]) for k in range(1, 7)] == [4, 25, 168, 1229, 9592, 78498]
    twins = equation.twin_count(is_prime)  # π₂(x): pairs (p, p + 2) with p ≤ x
    assert [int(twins[10**k]) for k in range(3, 7)] == [35, 205, 1224, 8169]


def test_ulam_spiral_is_a_walk_through_every_cell():
    side = 21
    n = torch.arange(1, side * side + 1)
    xy = tensors.ulam_xy(n, start=1)  # (N, 2)
    assert xy.abs().max() == side // 2
    assert len({tuple(p) for p in xy.tolist()}) == side * side  # a bijection onto the square
    step = (xy[1:] - xy[:-1]).abs().sum(dim=1)
    assert (step == 1).all()  # consecutive integers are neighbours
    assert xy[0].tolist() == [0, 0] and xy[1].tolist() == [1, 0]
    odd_squares = torch.arange(1, side, 2) ** 2  # (2k+1)² sits at the corner (k, −k)
    k = torch.arange(0, side // 2)
    assert (xy[odd_squares - 1] == torch.stack([k, -k], 1)).all()


def test_main_diagonal_of_the_spiral_is_a_quadratic():
    """On the spiral started at s, the diagonal x = y carries the numbers k² + k + s (Euler's for s = 41)."""
    k = torch.arange(-30, 31)
    m = k * k + k + 1  # positions on the standard spiral
    xy = tensors.ulam_xy(m + 40, start=41)
    assert (xy[:, 0] == xy[:, 1]).all()
    euler = equation.quadratic(torch.tensor([[1, 1, 41]]), torch.arange(0, 41))[0]
    is_prime = tensors.sieve(2000, CPU)
    assert is_prime[euler[:40]].all() and not is_prime[euler[40]]  # 40² + 40 + 41 = 41²


def test_sacks_spiral_puts_squares_on_the_positive_axis():
    n = torch.arange(1, 50, dtype=torch.float64) ** 2
    xy = tensors.sacks_xy(n)
    assert torch.allclose(xy[:, 1], torch.zeros(49, dtype=torch.float64), atol=1e-9)
    assert (xy[:, 0] > 0).all()


def test_goldbach_convolution_matches_brute_force():
    is_prime = tensors.sieve(4000, CPU)
    r = equation.goldbach_count(is_prime)  # r(N) = #{(p, q) ordered : p + q = N}
    for n in (4, 10, 100, 998, 1000, 3998):
        brute = sum(1 for p in range(2, n - 1) if _is_prime_slow(p) and _is_prime_slow(n - p))
        assert int(r[n]) == brute
    assert (r[4:4001:2] > 0).all()  # Goldbach holds in this range (verified to 4·10¹⁸, not proved)


def test_hardy_littlewood_constants():
    primes = torch.nonzero(tensors.sieve(10**6, CPU)).flatten()
    assert abs(equation.twin_constant(primes) - 0.6601618158) < 1e-5
    polys = torch.tensor([[1, 0, 1], [1, 1, 41]])
    c = equation.bateman_horn_constant(polys, primes[primes < 10**5])
    assert abs(c[0] - 1.3728134628) / 1.3728 < 0.01  # n² + 1 (Shanks)
    # Euler's n² + n + 41: Hardy–Littlewood's odd-prime product 3.3197732 times the factor 2 from p = 2 (always odd)
    assert abs(c[1] - 2 * 3.3197732) / 6.64 < 0.01
    assert equation.bateman_horn_constant(torch.tensor([[1, 1, 2]]), primes[:100])[0] == 0  # n² + n + 2 is always even


def test_twin_prediction_tracks_the_count():
    x = torch.tensor([1e5, 1e6], dtype=torch.float64)
    pred = equation.twin_prediction(x, 0.6601618158)
    ratio = torch.tensor([1224.0, 8169.0], dtype=torch.float64) / pred
    assert ((ratio - 1).abs() < 0.03).all()


def test_mean_prime_gap_is_log_x():
    """Prime number theorem: near x the average gap between primes is ln x."""
    primes = torch.nonzero(tensors.sieve(2 * 10**6, CPU)).flatten().double()
    near = primes[(primes > 1.5e6) & (primes < 2e6)]
    mean_gap = (near[-1] - near[0]) / (len(near) - 1)
    assert abs(mean_gap / math.log(1.75e6) - 1) < 0.02
