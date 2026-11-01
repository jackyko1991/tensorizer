"""Physics checks, written first (TDD): what any correct Lorenz-attractor code must satisfy.

Run with `pytest problems/others/lorenz_attractor/test.py`. Each test is a property of the dynamics, not an
implementation detail: the fixed points are stationary, phase-space volume contracts at the constant rate
σ + 1 + β, nearby orbits separate at the known Lyapunov exponent, and for ρ < 1 everything decays to the origin.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

LORENZ = (10.0, 28.0, 8.0 / 3.0)


def test_tensor_form_matches_explicit_equations():
    A, Q = tensors.lorenz_tensors(*LORENZ)
    x = torch.randn(20, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 15
    assert torch.allclose(equation.rhs(x, A, Q), equation.rhs_explicit(x, *LORENZ))


def test_fixed_points_have_zero_velocity():
    sigma, rho, beta = LORENZ
    A, Q = tensors.lorenz_tensors(*LORENZ)
    p = tensors.fixed_points(*LORENZ)
    c = math.sqrt(beta * (rho - 1))
    expected = torch.tensor([[0, 0, 0], [c, c, rho - 1], [-c, -c, rho - 1]], dtype=torch.float64)
    assert torch.allclose(p, expected)
    assert equation.rhs(p, A, Q).abs().max() < 1e-12  # equilibria: ẋ = 0
    assert len(tensors.fixed_points(10.0, 0.5, beta)) == 1  # pitchfork bifurcation: for ρ < 1 only the origin


def test_divergence_is_constant_volume_contraction():
    sigma, rho, beta = LORENZ
    A, Q = tensors.lorenz_tensors(*LORENZ)
    x = torch.randn(5, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 10
    div = equation.divergence(x, A, Q)
    # dissipative flow: phase-space volume shrinks as e^{-(σ+1+β)t} everywhere (Liouville with ∇·f < 0)
    assert torch.allclose(div, torch.full((5,), -(sigma + 1 + beta), dtype=torch.float64))
    # same as the trace of the Jacobian computed by autograd
    for xb in x:
        J = torch.autograd.functional.jacobian(lambda v: equation.rhs(v[None], A, Q)[0], xb)
        assert torch.trace(J).item() == pytest.approx(-(sigma + 1 + beta))


def test_lyapunov_exponent_is_about_point_nine():
    A, Q = tensors.lorenz_tensors(*LORENZ)
    center = solve.settle(A, Q)
    x0 = tensors.initial_ball(center, 1e-6, 8)
    times, traj = solve.solve(x0, A, Q, t_max=25.0)
    s = equation.spread(traj)
    lam, t0, t1, _ = solve.lyapunov_fit(times, s)
    assert 0.7 < lam < 1.1  # literature: largest Lyapunov exponent 0.906
    assert s[-1] > 1e6 * s[0]  # the butterfly effect: spread over the attractor


def test_below_rho_one_decays_to_origin():
    A, Q = tensors.lorenz_tensors(10.0, 0.5, 8.0 / 3.0)
    x0 = tensors.initial_ball(torch.tensor([5.0, -3.0, 10.0], dtype=torch.float64), 2.0, 6)
    _, traj = solve.solve(x0, A, Q, t_max=30.0)
    assert traj[-1].norm(dim=1).max() < 1e-3  # the origin is globally stable for ρ < 1
