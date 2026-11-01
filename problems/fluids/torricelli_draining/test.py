"""Physics checks, written first (TDD): what any correct tank-draining code must satisfy.

Run with `pytest problems/fluids/torricelli_draining/test.py`. With Torricelli's law v = √(2gh) the level follows the
exact quadratic h(t), and the drain time scales as (A/a)√h₀.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
G = 9.81


def test_matches_analytic_solution_and_drain_time():
    d = tensors.hole_diameters(3, 0.01, 0.03, CPU)
    a, A, h0 = tensors.circle_area(d), tensors.circle_area(0.4), 1.2
    T = equation.drain_time(a, A, h0, G)
    times, levels = solve.solve(torch.full((3,), h0), a, A, G, t_max=1.1 * T.max().item())
    exact = equation.analytic_height(times, a, A, h0, G)
    assert (levels - exact).abs().max() < 1e-5  # Torricelli: √h falls linearly in time
    assert torch.allclose(solve.numerical_drain_time(times, levels), T, rtol=3e-3)


def test_drain_time_scaling():
    a = torch.tensor([1e-4, 4e-4])
    T = equation.drain_time(a, 0.1, 1.0, G)
    assert (T[0] / T[1]).item() == pytest.approx(4.0)  # 4× the hole area → 4× faster
    assert (equation.drain_time(a, 0.1, 4.0, G) / T).tolist() == pytest.approx([2.0, 2.0])  # 4× height → 2× time
