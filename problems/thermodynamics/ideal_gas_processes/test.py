"""Physics checks, written first (TDD): what any correct ideal-gas-process code must satisfy.

Run with `pytest problems/thermodynamics/ideal_gas_processes/test.py`. The work on each path matches its closed
form, p V^γ is constant on the adiabat with Q = 0, and on the isotherm ΔU = 0 so Q = W (first law).
"""

import math

import torch

from . import equation, solve, tensors


def test_process_work_and_first_law():
    n, T0, ratio, gamma = 1.0, 300.0, 3.0, 5 / 3
    paths = tensors.processes(n, T0, ratio, gamma, points=2000)
    r = solve.solve(paths, n, gamma)
    W_iso, W_bar, W_cho, W_adi = r["W"].tolist()

    assert math.isclose(W_iso, n * equation.R * T0 * math.log(ratio), rel_tol=1e-5)  # isotherm: W = nRT ln(V2/V1)
    assert W_cho == 0.0  # dV = 0 on the isochore
    p1 = n * equation.R * T0 / tensors.V1
    assert math.isclose(W_bar, p1 * (ratio - 1) * tensors.V1, rel_tol=1e-10)  # isobar: W = p ΔV

    V, p = paths[3, :, 0], paths[3, :, 1]  # adiabat: p V^γ stays constant and Q = 0
    pVg = p * V**gamma
    assert torch.allclose(pVg, pVg[0].expand_as(pVg), rtol=1e-10)
    assert abs(r["Q"][3].item()) < 1e-4 * abs(W_adi)
    assert math.isclose(r["Q"][0].item(), W_iso, rel_tol=1e-8)  # isotherm: ΔU = 0, so Q = W
