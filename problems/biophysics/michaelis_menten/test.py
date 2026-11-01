"""Physics checks, written first (TDD): what any correct enzyme-kinetics code must satisfy.

Run with `pytest problems/biophysics/michaelis_menten/test.py`. Each test is a physical law: mass action conserves
enzyme and substrate, the rate law has the right limits, the reduced model matches the integrated Michaelis–Menten
law, and the quasi-steady state holds only for small enzyme.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def _mm_run(E0=0.1, t_end=30.0):
    k1, k_1, k2 = (k.to(F64) for k in tensors.rate_constants(CPU))
    S0 = tensors.substrate_concentrations(CPU).to(F64)
    y0 = tensors.initial_state(E0, S0, len(k1))
    K_m, V_max = equation.michaelis_constant(k1, k_1, k2), equation.max_velocity(k2, E0)
    times, full, S_q = solve.solve(y0, k1, k_1, k2, V_max, K_m, t_end)
    return times, full, S_q, S0, k1, k_1, k2, K_m, V_max


def test_conserves_enzyme_and_substrate():
    E0 = 0.5
    _, full, _, S0, *_ = _mm_run(E0=E0)
    E, S, ES, P = full.unbind(1)
    assert (E + ES - E0).abs().max() < 1e-10
    assert (S + ES + P - S0).abs().max() < 1e-9
    assert full.min() > -1e-12  # concentrations stay non-negative
    assert (P[1:] - P[:-1]).min() >= 0  # product only accumulates


def test_rate_law_limits():
    V_max, K_m = torch.tensor([2.0], dtype=F64), torch.tensor([5.0], dtype=F64)
    v = equation.mm_rate(torch.tensor([5.0, 1e9, 1e-6], dtype=F64), V_max, K_m)[0]
    assert math.isclose(v[0].item(), 1.0, rel_tol=1e-12)  # half speed at S = K_m
    assert math.isclose(v[1].item(), 2.0, rel_tol=1e-6)  # saturation at V_max
    assert math.isclose(v[2].item(), 2.0 / 5.0 * 1e-6, rel_tol=1e-6)  # linear regime v ≈ (V_max/K_m) S


def test_reduced_model_matches_integrated_law():
    times, _, S_q, S0, *_, K_m, V_max = _mm_run(t_end=30.0)
    for i in (100, len(times) // 2, len(times) - 1):
        t_exact = equation.integrated_mm_time(S_q[i], S0, V_max, K_m)  # (B, C)
        assert (t_exact - times[i]).abs().max() < 1e-6


def test_qssa_valid_for_small_enzyme_and_fails_for_large():
    def gap(E0):
        _, full, S_q, S0, *_ = _mm_run(E0=E0, t_end=20.0)
        return ((full[:, 3] - (S0 - S_q)).abs() / S0).max().item()

    small, large = gap(0.01), gap(2.0)
    assert small < 2e-3
    assert large > 10 * small


def test_lineweaver_burk_recovers_constants():
    # exact Michaelis–Menten data: the fit is exact
    S = torch.tensor([[1.0, 2.0, 5.0, 10.0, 50.0]], dtype=F64)
    v = equation.mm_rate(S, torch.tensor([3.0], dtype=F64), torch.tensor([7.0], dtype=F64))
    K_fit, V_fit = equation.lineweaver_burk(S, v)
    assert math.isclose(K_fit.item(), 7.0, rel_tol=1e-10) and math.isclose(V_fit.item(), 3.0, rel_tol=1e-10)

    # initial rates measured on the full mass-action simulation
    _, full, _, _, _, _, k2, K_m, V_max = _mm_run(t_end=10.0)
    S_v, v0 = solve.initial_rate(full, k2)
    K_fit, V_fit = equation.lineweaver_burk(S_v, v0)
    assert torch.allclose(K_fit, K_m, rtol=1e-2)
    assert torch.allclose(V_fit, V_max, rtol=1e-2)
