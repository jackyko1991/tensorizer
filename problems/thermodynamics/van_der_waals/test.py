"""Physics checks, written first (TDD): what any correct van der Waals code must satisfy.

Run with `pytest problems/thermodynamics/van_der_waals/test.py`. The critical point is an inflection of the critical
isotherm at the closed-form (T_c, V_c, p_c), Maxwell's equal-area rule fixes the coexistence pressure, and the
law of corresponding states gives p ≈ 0.383 p_c at 0.8 T_c.
"""

import math

import torch

from . import equation, solve, tensors


def test_critical_point():
    a, b = tensors.GASES["CO2"]
    n = 2.0
    T_c, V_c, p_c = equation.critical_point(n, a, b)
    assert math.isclose(V_c, 3 * n * b) and math.isclose(p_c, a / (27 * b**2))
    assert math.isclose(T_c, 8 * a / (27 * equation.R * b))

    # the critical isotherm passes through (V_c, p_c) with zero slope and zero curvature
    h = 1e-3 * V_c
    V = torch.tensor([V_c - h, V_c, V_c + h], dtype=torch.float64)
    p = equation.pressure(V, torch.tensor([T_c], dtype=torch.float64), n, a, b)[0]
    assert math.isclose(p[1].item(), p_c, rel_tol=1e-10)
    assert abs((p[2] - p[0]).item() / (2 * h)) * V_c / p_c < 1e-5  # ∂p/∂V = 0 at the critical point
    assert abs((p[2] - 2 * p[1] + p[0]).item() / h**2) * V_c**2 / p_c < 1e-3  # ∂²p/∂V² = 0


def test_maxwell_equal_areas():
    a, b = tensors.GASES["water"]
    n = 1.0
    T_c, V_c, p_c = equation.critical_point(n, a, b)
    T = tensors.temperatures(T_c, 1.1, 0.75, 8)
    V = tensors.volumes(V_c)
    r = solve.solve(T, V, T_c, V_c, n, a, b)
    below = T < T_c
    assert torch.isnan(r["p_star"][~below]).all()  # no liquid–gas coexistence above T_c

    for t in torch.nonzero(below).flatten().tolist():
        p_star, V_l, V_g = r["p_star"][t].item(), r["V_l"][t].item(), r["V_g"][t].item()
        assert V_l < V_c < V_g and 0 < p_star < p_c
        # independent check on a fine grid: the signed area between curve and line vanishes
        V_fine = torch.linspace(V_l, V_g, 20001, dtype=torch.float64)
        gap = equation.pressure(V_fine, T[t : t + 1], n, a, b)[0] - p_star
        signed, total = torch.trapezoid(gap, V_fine), torch.trapezoid(gap.abs(), V_fine)
        assert abs(signed.item()) < 1e-4 * total.item()  # Maxwell construction: equal Gibbs energies
        assert abs(gap[0].item()) < 1e-6 * p_c and abs(gap[-1].item()) < 1e-6 * p_c  # line ends on the curve

    # reduced vapour pressure of any van der Waals fluid at T = 0.8 T_c is about 0.383 p_c
    p_08 = solve.coexistence_pressure(torch.tensor([0.8 * T_c], dtype=torch.float64), V, V_c, n, a, b)
    assert math.isclose(p_08.item() / p_c, 0.3834, rel_tol=2e-3)  # law of corresponding states
