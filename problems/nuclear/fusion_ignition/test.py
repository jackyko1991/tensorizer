"""Physics checks, written first (TDD): what any correct fusion-ignition code must satisfy.

Run with `pytest problems/nuclear/fusion_ignition/test.py`. Each test is a physical law, not an implementation
detail: the DT reactivity has the measured values and peak, the ideal ignition temperature is about 4.3 keV, heating
balances losses on the boundary and more losses shrink the ignition region.
"""

import pytest
import torch

from . import equation, solve, tensors


def test_dt_reactivity_values_and_peak():
    T = torch.tensor([10.0, 20.0], dtype=torch.float64)
    sv = equation.sigma_v(T)
    assert sv[0].item() == pytest.approx(1.1e-22, rel=0.05)  # ≈ 1.1e-22 m³/s near 10 keV
    assert sv[1].item() == pytest.approx(4.3e-22, rel=0.05)
    grid = torch.linspace(5.0, 100.0, 9501, dtype=torch.float64)
    s = equation.sigma_v(grid)
    assert 50.0 < grid[s.argmax()].item() < 80.0
    assert 7e-22 < s.max().item() < 1e-21  # peak of order 1e-21 m³/s


def test_ideal_ignition_temperature():
    assert 4.0 < equation.ideal_ignition_temperature() < 4.7  # alpha heating = bremsstrahlung at ≈ 4.3 keV


def test_balance_boundary_moves_out_when_losses_grow():
    T = torch.linspace(6.0, 30.0, 25, dtype=torch.float64)
    base = solve.balance_boundary(T, 100.0, 1.0)
    lossy = solve.balance_boundary(T, 100.0, 1.5)
    assert torch.isfinite(base).all() and torch.isfinite(lossy).all()
    assert torch.all(lossy > base)
    # at the boundary heating balances losses
    margin = equation.heating_balance(T, base, 100.0, 1.0)
    scale = equation.powers(T, base, torch.zeros_like(T), 100.0)["alpha"]
    assert torch.all(margin.abs() / scale < 1e-6)  # power balance on the boundary


def test_integrated_ignition_region_shrinks_with_more_losses():
    t0, r0 = tensors.initial_grid(8, 10, (4.0, 16.0), (0.05, 1.5))
    y0 = tensors.initial_state(t0, r0)
    results = {}
    for loss in (1.0, 1.6):
        _, states = solve.integrate(y0, 100.0, loss, 2.0, 120.0, 0.25, 20)
        assert torch.isfinite(states).all()
        ign = solve.ignited(states)
        results[loss] = (ign, solve.min_ignition_rho_r(t0, r0, ign))
    ign1, first1 = results[1.0]
    ign2, first2 = results[1.6]
    assert ign1.sum() > ign2.sum() > 0
    assert torch.all(ign2 <= ign1)  # anything that ignites with more losses also ignites with fewer
    both = torch.isfinite(first1) & torch.isfinite(first2)
    assert both.any() and torch.all(first2[both] >= first1[both])
    assert not ign1.view(8, 10)[0].any()  # 4 keV starts never ignite: below the ignition temperature range
