"""Physics checks, written first (TDD): what any correct special-relativity code must satisfy.

Run with `pytest problems/astro/minkowski_diagram/test.py`. Each test is a physical law, not an implementation
detail: the interval is invariant under boosts, boosts compose by relativistic velocity addition and the travelling
twin ages less.
"""

import math

import pytest
import torch

from . import equation, solve, tensors


def test_interval_is_invariant_under_boosts():
    eta = equation.metric()
    events = torch.randn(50, 2, dtype=torch.float64)
    for beta in (-0.9, -0.3, 0.5, 0.99):
        boosted = equation.transform(equation.boost(torch.tensor(beta)), events)
        # Lorentz invariance of s²
        assert torch.allclose(equation.interval(boosted, eta), equation.interval(events, eta), atol=1e-9)


def test_boost_composition_is_velocity_addition():
    b1, b2 = torch.tensor(0.6, dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64)
    combined = equation.boost(b1) @ equation.boost(b2)
    assert torch.allclose(combined, equation.boost(equation.velocity_addition(b1, b2)), atol=1e-12)
    assert equation.velocity_addition(b1, b2) < 1  # nothing exceeds c


def test_twin_paradox_ages():
    v, L = 0.8, 4.0
    home, traveller = tensors.twin_world_lines(v, L)
    sol = solve.solve(home, traveller, tensors.frame_grid(1, -1, 1), beta_view=0.3)
    assert sol["tau_home"][-1].item() == pytest.approx(2 * L / v)
    # time dilation: τ = t √(1 − v²)
    assert sol["tau_trav"][-1].item() == pytest.approx(2 * L / v * math.sqrt(1 - v**2), rel=1e-9)
