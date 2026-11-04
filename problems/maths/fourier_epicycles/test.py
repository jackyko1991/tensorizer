"""Maths checks, written first (TDD): what any correct epicycle (complex DFT) code must satisfy.

Run with `pytest problems/maths/fourier_epicycles/test.py`. Each test is an exact property of the discrete Fourier
transform: it is invertible, it preserves energy (Parseval), all N circles retrace the samples, dropping the smallest
circles first makes the error fall monotonically, a circle is one term, and a time shift is a phase.
"""

import math

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")


def _curve(name="logo", n=256):
    return tensors.shape(name, n, CPU)


@pytest.mark.parametrize("name", tensors.SHAPES)
def test_dft_idft_round_trip_and_matches_fft(name):
    z = _curve(name)
    c, k = equation.dft(z)
    assert torch.allclose(equation.idft(c, k, tensors.times(z.shape[0], CPU)), z, atol=1e-10)
    fft = torch.fft.fft(z) / z.shape[0]  # torch's FFT uses the same sign convention, frequencies 0, 1, ..., -1
    assert torch.allclose(c, fft[k % z.shape[0]], atol=1e-12)


@pytest.mark.parametrize("name", tensors.SHAPES)
def test_parseval(name):
    z = _curve(name)
    c, _ = equation.dft(z)
    assert equation.mean_power(z).item() == pytest.approx((c.abs() ** 2).sum().item(), rel=1e-12)


def test_all_terms_reproduce_the_samples_exactly():
    z = _curve("star", 200)
    c, k = equation.dft(z)
    order = equation.by_radius(c)
    s = equation.partial_sums(c, k, order, torch.tensor([z.shape[0]]), tensors.times(z.shape[0], CPU))  # (1, N)
    assert torch.allclose(s[0], z, atol=1e-10)
    tips = equation.chain(c, k, order, tensors.times(z.shape[0], CPU))  # (T, K + 1) circle centres, last = pen
    assert torch.allclose(tips[:, -1], z, atol=1e-10)


def test_error_falls_monotonically_when_sorted_by_radius():
    z = _curve("logo", 300)
    c, k = equation.dft(z)
    order = equation.by_radius(c)
    m = torch.arange(1, z.shape[0] + 1)
    err = equation.sample_error(z, equation.partial_sums(c, k, order, m, tensors.times(z.shape[0], CPU)))  # (M,)
    assert torch.all(err[1:] <= err[:-1] + 1e-12)
    assert err[-1] < 1e-20
    # Parseval: the error with M circles is exactly the energy of the circles left out
    left_out = torch.flip(torch.cumsum(torch.flip(c[order].abs() ** 2, [0]), 0), [0])  # (N,) energy from rank M on
    assert torch.allclose(err[:-1], left_out[1:], rtol=1e-8, atol=1e-14)


def test_a_circle_needs_one_term():
    n = 128
    t = tensors.times(n, CPU)
    z = 1.7 * torch.exp(2j * math.pi * 3 * t + 0.4j)  # radius 1.7, three turns, phase 0.4
    c, k = equation.dft(z)
    big = c.abs() > 1e-10
    assert big.sum() == 1 and k[big].item() == 3
    assert c[big].abs().item() == pytest.approx(1.7) and c[big].angle().item() == pytest.approx(0.4)


def test_time_shift_is_a_phase():
    z = _curve("heart", 160)
    n, s = z.shape[0], 23
    c, k = equation.dft(z)
    c_shift, _ = equation.dft(torch.roll(z, s))  # z_{n - s}
    assert torch.allclose(c_shift, c * torch.exp(-2j * math.pi * k * s / n), atol=1e-12)
    assert torch.allclose(c_shift.abs(), c.abs(), atol=1e-12)  # the radii do not change: only the start point moves


def test_heart_is_exactly_eight_circles():
    c, k = equation.dft(_curve("heart", 256))  # 16 sin³s = 12 sin s − 4 sin 3s: frequencies ±1, ±2, ±3, ±4 only
    big = c.abs() > 1e-10
    assert big.sum() == 8
    assert sorted(k[big].tolist()) == [-4, -3, -2, -1, 1, 2, 3, 4]


def test_polygons_are_closed_and_evenly_sampled():
    for name in ("logo", "star"):
        z = _curve(name, 400)
        step = (torch.roll(z, -1) - z).abs()  # includes the closing segment back to the start
        assert step.max() < 3 * step.mean()  # resampled by arc length: no big jumps
        assert z.abs().max() <= 1.0 + 1e-9  # scaled into the unit disc
