import math

import pytest
import torch

from problems.others.lorenz_attractor import equation as lz_eq
from problems.others.lorenz_attractor import solve as lz_solve
from problems.others.lorenz_attractor import tensors as lz_t

CPU = torch.device("cpu")

# --- Lorenz attractor ---

LORENZ = (10.0, 28.0, 8.0 / 3.0)


def test_lorenz_tensor_form_matches_explicit_equations():
    A, Q = lz_t.lorenz_tensors(*LORENZ)
    x = torch.randn(20, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)) * 15
    assert torch.allclose(lz_eq.rhs(x, A, Q), lz_eq.rhs_explicit(x, *LORENZ))


def test_lorenz_fixed_points_have_zero_velocity():
    sigma, rho, beta = LORENZ
    A, Q = lz_t.lorenz_tensors(*LORENZ)
    p = lz_t.fixed_points(*LORENZ)
    c = math.sqrt(beta * (rho - 1))
    expected = torch.tensor([[0, 0, 0], [c, c, rho - 1], [-c, -c, rho - 1]], dtype=torch.float64)
    assert torch.allclose(p, expected)
    assert lz_eq.rhs(p, A, Q).abs().max() < 1e-12
    assert len(lz_t.fixed_points(10.0, 0.5, beta)) == 1  # for ρ < 1 only the origin


def test_lorenz_divergence_is_constant_volume_contraction():
    sigma, rho, beta = LORENZ
    A, Q = lz_t.lorenz_tensors(*LORENZ)
    x = torch.randn(5, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 10
    div = lz_eq.divergence(x, A, Q)
    assert torch.allclose(div, torch.full((5,), -(sigma + 1 + beta), dtype=torch.float64))
    # same as the trace of the Jacobian computed by autograd
    for xb in x:
        J = torch.autograd.functional.jacobian(lambda v: lz_eq.rhs(v[None], A, Q)[0], xb)
        assert torch.trace(J).item() == pytest.approx(-(sigma + 1 + beta))


def test_lorenz_lyapunov_exponent_is_about_point_nine():
    A, Q = lz_t.lorenz_tensors(*LORENZ)
    center = lz_solve.settle(A, Q)
    x0 = lz_t.initial_ball(center, 1e-6, 8)
    times, traj = lz_solve.solve(x0, A, Q, t_max=25.0)
    s = lz_eq.spread(traj)
    lam, t0, t1, _ = lz_solve.lyapunov_fit(times, s)
    assert 0.7 < lam < 1.1  # literature: 0.906
    assert s[-1] > 1e6 * s[0]  # the butterfly effect: spread over the attractor


def test_lorenz_below_rho_one_decays_to_origin():
    A, Q = lz_t.lorenz_tensors(10.0, 0.5, 8.0 / 3.0)
    x0 = lz_t.initial_ball(torch.tensor([5.0, -3.0, 10.0], dtype=torch.float64), 2.0, 6)
    _, traj = lz_solve.solve(x0, A, Q, t_max=30.0)
    assert traj[-1].norm(dim=1).max() < 1e-3
