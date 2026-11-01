"""Physics checks, written first (TDD): what any correct Hawking-radiation code must satisfy.

Run with `pytest problems/astro/hawking_radiation/test.py`. Each test is a physical law, not an implementation
detail: T_H ∝ 1/M (6.2e-8 K for the Sun), the Planck spectrum obeys Wien's law, P ∝ 1/M² and the integrated lifetime
matches t ∝ M³.
"""

import math

import pytest
import torch

from . import equation, solve


def test_temperature_of_the_sun_and_inverse_mass_law():
    M = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float64) * equation.M_SUN
    T = equation.hawking_temperature(M)
    assert T[0].item() == pytest.approx(6.17e-8, rel=5e-3)
    assert torch.allclose(T * M, (T * M)[0].expand(3), rtol=1e-12)  # T_H ∝ 1/M


def test_wien_peak_of_planck_spectrum():
    T = torch.tensor([1e3, 1e9], dtype=torch.float64)
    lam = torch.logspace(-16, -2, 200001, dtype=torch.float64)
    B = equation.planck(lam, T)  # (2, L)
    peak = lam[B.argmax(dim=1)]
    # Wien's displacement law: λ_max T = b
    assert torch.allclose(peak * T, torch.full_like(T, equation.WIEN_B), rtol=1e-3)


def test_integrated_lifetime_matches_formula():
    M0 = torch.tensor([1e5, 1e11], dtype=torch.float64)
    times, masses = solve.evaporate(M0, fraction=0.005)
    assert torch.allclose(times[-1], equation.lifetime(M0), rtol=1e-6)  # lifetime t ∝ M0³
    # M³ falls linearly in time: t = t_life(M0) − t_life(M) at every step
    remaining = equation.lifetime(M0) - equation.lifetime(masses)
    assert ((times - remaining).abs() / equation.lifetime(M0)).max() < 1e-6


def test_power_scales_as_inverse_mass_squared():
    M = torch.tensor([1e10, 2e10], dtype=torch.float64)
    P = equation.power(M)
    assert (P[0] / P[1]).item() == pytest.approx(4.0)  # P ∝ 1/M²
    expected = equation.HBAR * equation.C**6 / (15360 * math.pi * equation.G**2 * M**2)
    assert torch.allclose(P, expected, rtol=1e-6)
