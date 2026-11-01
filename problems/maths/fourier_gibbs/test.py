"""Physics checks, written first (TDD): what any correct Fourier series code must satisfy.

Run with `pytest problems/maths/fourier_gibbs/test.py`. Each test is a mathematical invariant, not an implementation
detail: the square-wave coefficients are 4/(πn), Parseval's identity holds, the Gibbs overshoot tends to 8.95% of the
jump, and the L² error decays at the rate set by the smoothness of the wave.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def setup(n_x: int = 8192, n_max: int = 101):
    x, w = tensors.grid(n_x, CPU)
    f = tensors.waves(x)
    b = tensors.basis(x, n_max)
    return x, w, f, b, equation.coefficients(f, b, w)


def test_square_wave_coefficients_are_4_over_pi_n_for_odd_n():
    _, _, _, _, c = setup()
    n = torch.arange(1, c.shape[1] + 1, dtype=torch.float64)
    expected = torch.where(n % 2 == 1, 4 / (math.pi * n), torch.zeros_like(n))
    assert torch.allclose(c[0], expected, atol=1e-5)
    assert torch.allclose(c, tensors.exact_coefficients(c.shape[1], CPU), atol=1e-5)


def test_gibbs_overshoot_tends_to_8_95_percent_of_the_jump():
    _, _, f, b, c = setup()
    s = torch.stack([equation.partial_sum(b, c, n) for n in (25, 51, 101)])
    over = solve.overshoot(s, f, tensors.JUMP)
    gibbs = 0.08949  # (1/π) Si(π) − 1/2
    assert torch.allclose(over[:, :2], torch.full_like(over[:, :2], gibbs), atol=2e-3)  # square and sawtooth
    assert over[-1, 2] < 1e-3  # the continuous triangle wave has no overshoot


def test_parseval_energy_equals_sum_of_squared_coefficients():
    _, w, f, _, _ = setup()
    exact = tensors.exact_coefficients(200000, CPU)
    # (1/π)∫ f² dx: 2 for the square wave, 2/3 for the sawtooth and the triangle wave
    assert torch.allclose((exact**2).sum(dim=1), torch.tensor([2.0, 2 / 3, 2 / 3], dtype=torch.float64), atol=1e-5)
    assert torch.allclose(equation.energy(f, w), torch.tensor([2.0, 2 / 3, 2 / 3], dtype=torch.float64), atol=1e-6)
    # the squared L² error of a partial sum is the energy of the missing coefficients
    _, w, f, b, c = setup(n_max=60)
    err = equation.l2_error_sq(f, equation.partial_sum(b, c, 20), w)
    assert torch.allclose(err, equation.energy(f, w) - (c[:, :20] ** 2).sum(dim=1), atol=1e-9)


def test_l2_error_decays_like_1_over_n_for_jumps_and_1_over_n_cubed_for_a_kink():
    _, w, f, b, c = setup(n_x=4096, n_max=100)
    s = solve.all_partial_sums(b, c)
    p = solve.decay_rate(solve.errors(f, s, w))
    assert torch.allclose(p, torch.tensor([1.0, 1.0, 3.0], dtype=torch.float64), atol=0.1)
