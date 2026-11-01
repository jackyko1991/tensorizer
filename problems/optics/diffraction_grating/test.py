"""Physics checks, written first (TDD): what any correct diffraction-grating code must satisfy.

Run with `pytest problems/optics/diffraction_grating/test.py`. The phasor sum equals the closed-form N-slit pattern,
principal maxima sit at d sin θ = mλ, and there are N − 2 secondary maxima between orders.
"""

import torch

from . import equation, solve


def test_matches_closed_form_and_maxima():
    lam, d = 0.5, 2.0
    k = equation.wavenumber(lam)
    sin_theta = torch.linspace(-0.9, 0.9, 20001, dtype=torch.float64)
    for n in (2, 5, 12):
        e = equation.array_factor(n, sin_theta, k, d)
        assert torch.allclose((e.conj() * e).real, equation.closed_form(sin_theta, n, k, d), atol=1e-8)
    orders = torch.tensor(solve.principal_maxima(lam, d, 0.9), dtype=torch.float64)
    assert torch.allclose(orders, torch.arange(-3, 4, dtype=torch.float64) * lam / d)  # d sin θ = m λ
    e = equation.array_factor(7, orders, k, d)
    assert torch.allclose(e.abs(), torch.ones_like(orders), atol=1e-9)  # all phasors aligned: |E| = 1


def test_has_n_minus_2_secondary_maxima():
    lam, d = 0.5, 2.0
    k = equation.wavenumber(lam)
    sin_theta = torch.linspace(0.0, lam / d, 40001, dtype=torch.float64)[1:-1]  # strictly between orders 0 and 1
    for n in (2, 3, 6, 10, 20):
        e = equation.array_factor(n, sin_theta, k, d)
        i = (e.conj() * e).real
        peaks = (i[1:-1] > i[:-2]) & (i[1:-1] > i[2:])
        assert peaks.sum().item() == n - 2
