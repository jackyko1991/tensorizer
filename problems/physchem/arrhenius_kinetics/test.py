"""Physics checks, written first (TDD): what any correct rate-law and Arrhenius code must satisfy.

Run with `pytest problems/physchem/arrhenius_kinetics/test.py`. Each test is a physical law, not an implementation
detail: the integrated rate laws match the analytic solutions, half-lives reveal the order, and ln k is linear in 1/T.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def _runs(Ea=75e3, A=1e11):
    T = tensors.temperatures(290.0, 340.0, CPU)
    k = equation.rate_constant(T, A, Ea)
    n = tensors.orders(CPU)
    t_end = tensors.run_lengths(k, n)
    tau, traj = solve.integrate_runs(tensors.initial_state(3, len(T), CPU), k, n, t_end)
    return T, k, n, t_end, tau, traj


def test_integrated_rate_laws_match_analytic():
    _, k, _, t_end, tau, traj = _runs()
    for o in range(3):
        t = tau[:, None] * t_end[o][None, :]  # (N+1, T) real times
        exact = equation.integrated_law(t, tensors.A0, k[None, :], o)
        assert (traj[:, o] - exact).abs().max() < 1e-8, f"order {o}"  # exact solution of d[A]/dt = -k [A]^n
    assert torch.allclose(traj[-1], torch.full_like(traj[-1], 0.1), atol=1e-8)  # each run ends at 90 % conversion


def test_half_life_method_gives_the_order():
    _, _, _, t_end, tau, traj = _runs()
    t50 = solve.crossing_time(tau, traj, t_end, 0.5)
    t25 = solve.crossing_time(tau, traj, t_end, 0.25)
    n_est = equation.order_from_half_lives(t50, t25 - t50)
    expected = torch.tensor([0.0, 1.0, 2.0], dtype=F64)[:, None].expand_as(n_est)
    assert torch.allclose(n_est, expected, atol=1e-3)  # successive half-lives scale as 2^(n-1)


def test_arrhenius_fit_recovers_Ea_and_A_exactly_from_exact_k():
    T = torch.linspace(280.0, 360.0, 7, dtype=F64)
    for Ea, A in ((50e3, 1e8), (75e3, 1e11), (100e3, 1e14)):
        Ea_fit, A_fit = equation.arrhenius_fit(T, equation.rate_constant(T, A, Ea))
        assert math.isclose(Ea_fit, Ea, rel_tol=1e-9)  # Arrhenius: ln k = ln A - Ea / RT is exactly linear in 1/T
        assert math.isclose(A_fit, A, rel_tol=1e-6)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_noisy_experiment_finds_order_and_activation_energy(order):
    T, k, _, t_end, tau, traj = _runs(Ea=80e3, A=3e12)
    tau_s, clean = solve.sample(tau, traj, order)
    data = tensors.measure(clean, 0.01)
    times = tau_s[None, :] * t_end[order][:, None]
    beta, R2 = solve.fit_linearizations(times, data)
    assert int(R2.mean(1).argmax()) == order  # the right plot is the straightest
    k_fit = (-1.0 if order < 2 else 1.0) * beta[order, :, 1]
    assert ((k_fit - k) / k).abs().max() < 0.06
    Ea_fit, A_fit = equation.arrhenius_fit(T, k_fit)
    assert abs(Ea_fit - 80e3) / 80e3 < 0.03
    assert abs(math.log(A_fit / 3e12)) < 1.0  # ln A is a long extrapolation to 1/T = 0


def test_least_squares_matches_lstsq():
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(4, 30, generator=gen, dtype=F64)
    y = 2.0 - 3.0 * x + 0.1 * torch.randn(4, 30, generator=gen, dtype=F64)
    beta = equation.least_squares(x, y)
    for b in range(4):
        X = torch.stack([torch.ones(30, dtype=F64), x[b]], 1)
        ref = torch.linalg.lstsq(X, y[b, :, None]).solution[:, 0]
        assert torch.allclose(beta[b], ref, atol=1e-10)
