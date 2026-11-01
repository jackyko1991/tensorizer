"""Physics checks, written first (TDD): what any correct radioactive-decay code must satisfy.

Run with `pytest problems/nuclear/radioactive_decay/test.py`. Each test is a physical law, not an implementation
detail: the half-life halves the exact count and the Monte Carlo mean follows N0 e^(−λt) within binomial noise, with
decayed nuclei never coming back.
"""

import math

import torch

from . import equation, solve


def test_half_life_halves_the_exact_count():
    lam = equation.decay_constant(3.0)
    assert math.isclose(math.log(2) / lam, 3.0)
    t = torch.tensor([0.0, 3.0, 6.0], dtype=torch.float64)
    # N(t) = N0 2^(−t/T½)
    assert torch.allclose(equation.exact_count(1000, lam, t), torch.tensor([1000.0, 500.0, 250.0], dtype=torch.float64))


def test_monte_carlo_mean_matches_exponential():
    lam, dt, n_steps = equation.decay_constant(2.0), 0.1, 60
    alive0 = torch.ones(20, 30, 30, dtype=torch.bool)  # 20 repeated experiments of a 30 × 30 grid
    history = solve.simulate(alive0, lam, dt, n_steps, seed=3)  # (T, 20, 30, 30)
    counts = equation.count(history).double()  # (T, 20)
    t = torch.arange(n_steps + 1, dtype=torch.float64) * dt
    n0 = 900
    survive = torch.exp(-lam * t)
    sigma_mean = torch.sqrt(n0 * survive * (1 - survive) / counts.shape[1])  # binomial σ of the mean of 20 runs
    deviation = (counts.mean(dim=1) - n0 * survive).abs()
    assert torch.all(deviation <= 4 * sigma_mean + 1e-9)  # mean follows N0 e^(−λt) within 4σ
    assert torch.all(history[1:] <= history[:-1])  # decayed nuclei never come back
