"""Physics checks, written first (TDD): what any correct Monte Carlo π code must satisfy.

Run with `pytest problems/others/monte_carlo_pi/test.py`. Each test is a law of probability, not an implementation
detail: the dart estimate converges to π with the binomial standard error σ = 4√(p(1−p)/N), the running estimate
is a prefix mean, and Buffon's needle gives π too.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_darts_converge_with_the_binomial_error():
    est = solve.darts_estimate(tensors.darts(64, 20000, 0, CPU))
    sd = solve.spread(est)
    assert abs(est[:, -1].mean().item() - math.pi) < 4 * sd[-1].item() / 8  # law of large numbers, mean of 64: σ/8
    ratio = sd[-1] / equation.standard_error(torch.tensor(20000.0, dtype=torch.float64))
    assert 0.7 < ratio.item() < 1.3  # central limit theorem: spread = binomial standard error ∝ 1/√N


def test_running_estimate_is_a_prefix_mean():
    h = torch.tensor([[1.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    expected = torch.tensor([[4.0, 2.0, 8 / 3, 3.0]], dtype=torch.float64)  # 4 × (hits so far) / (darts so far)
    assert torch.allclose(equation.running_estimate(h, 4.0), expected)


def test_buffon_needle_estimates_pi():
    y, phi = tensors.needles(16, 50000, 0, CPU)
    assert abs(solve.needle_estimate(y, phi, 0.8).mean().item() - math.pi) < 0.02  # Buffon: P(cross) = 2ℓ/(πd)
