"""Physics checks, written first (TDD): what any correct physics-informed neural network must satisfy.

Run with `pytest problems/others/pinn_oscillator/test.py`. Each test is a physical or mathematical law, not an
implementation detail: the exact damped-oscillator solution has zero residual m ẍ + μ ẋ + k x = 0, the trained
PINN matches it and extrapolates beyond the data, the inverse problem recovers μ, and without physics it fails.
"""

import pytest
import torch

from . import equation, solve, tensors
from .main import SEED, T_MAX, WIDTHS, M

CPU = torch.device("cpu")
K, MU = 225.0, 3.0  # the defaults in meta.yaml


def _rel_err(x, ref):
    return (torch.norm(x - ref) / torch.norm(ref)).item()


def _setup():
    t_d, x_d = tensors.observations(10, 0.4 * T_MAX, 0.02, M, MU, K, SEED, CPU)
    t_c = tensors.collocation_points(40, T_MAX, CPU)
    t_plot = tensors.plot_grid(T_MAX, CPU)
    return t_d, x_d, t_c, t_plot


def _train(lam_phys, inverse, n_iter, mu_start=1.0, n_snapshots=5):
    t_d, x_d, t_c, t_plot = _setup()
    weights = tensors.init_weights(WIDTHS, SEED, CPU)
    out = solve.train(
        weights, mu_start if inverse else MU, inverse, t_d, x_d, t_c, M, K, lam_phys, n_iter, 0.01, t_plot,
        n_snapshots=n_snapshots,
    )  # fmt: skip
    return out, t_plot, t_d


@pytest.fixture(scope="module")
def inverse_run():
    return _train(1e-4, True, 4000)


@pytest.fixture(scope="module")
def plain_run():
    return _train(0.0, False, 4000)


def test_einsum_layer_equals_linear():
    torch.manual_seed(0)
    h, W, b = torch.randn(7, 5), torch.randn(3, 5), torch.randn(3)
    assert torch.allclose(equation.layer(h, W, b), torch.nn.functional.linear(h, W, b), atol=1e-6)
    assert torch.allclose(equation.layer(h, W, b), h @ W.T + b, atol=1e-6)
    # the whole network equals the same layers built from torch.nn
    weights = tensors.init_weights([1, 8, 8, 1], 3, CPU)
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    )
    with torch.no_grad():
        for lin, (Wl, bl) in zip([net[0], net[2], net[4]], weights):
            lin.weight.copy_(Wl)
            lin.bias.copy_(bl)
        t = torch.linspace(0, 1, 11)
        assert torch.allclose(equation.network(t, weights), net(t[:, None])[:, 0], atol=1e-6)


def test_residual_autograd_matches_known_functions():
    t = torch.linspace(0, 1, 21, dtype=torch.float64).requires_grad_(True)
    # polynomial x = t³:  r = m·6t + μ·3t² + k·t³
    x = t**3
    x_t = equation.time_derivative(x, t)
    x_tt = equation.time_derivative(x_t, t)
    r = equation.residual(x, x_t, x_tt, M, MU, K)
    tt = t.detach()
    assert torch.allclose(r, M * 6 * tt + MU * 3 * tt**2 + K * tt**3, atol=1e-10)
    # the exact solution has zero residual and the right initial conditions
    x = equation.exact_solution(t, M, MU, K)
    x_t = equation.time_derivative(x, t)
    x_tt = equation.time_derivative(x_t, t)
    r = equation.residual(x, x_t, x_tt, M, MU, K)
    assert r.abs().max() < 1e-9 * K  # the analytic solution obeys m ẍ + μ ẋ + k x = 0
    assert abs(x[0].item() - 1.0) < 1e-12 and abs(x_t[0].item()) < 1e-10  # x(0) = 1, ẋ(0) = 0


def test_network_derivatives_match_finite_differences():
    weights = [(W.detach().double(), b.detach().double()) for W, b in tensors.init_weights(WIDTHS, 1, CPU)]
    t = torch.linspace(0.05, 0.95, 13, dtype=torch.float64)
    x, x_t, x_tt = equation.network_and_derivatives(t, weights)
    h = 1e-4
    xp, x0, xm = (equation.network(t + s, weights) for s in (h, 0.0, -h))
    assert torch.allclose(x, x0)
    assert torch.allclose(x_t, (xp - xm) / (2 * h), atol=1e-6)
    assert torch.allclose(x_tt, (xp - 2 * x0 + xm) / h**2, atol=1e-4)
    # and therefore the residual agrees with its finite-difference version
    r_fd = equation.residual(x0, (xp - xm) / (2 * h), (xp - 2 * x0 + xm) / h**2, M, MU, K)
    assert torch.allclose(equation.residual(x, x_t, x_tt, M, MU, K), r_fd, atol=1e-3)


def test_matches_exact_solution_and_extrapolates(inverse_run):
    out, t_plot, t_d = inverse_run
    exact = equation.exact_solution(t_plot, M, MU, K)
    x = out["snapshots"][-1]
    extrap = t_plot > t_d.max()
    assert _rel_err(x, exact) < 0.06  # matches the analytic damped oscillator
    assert _rel_err(x[extrap], exact[extrap]) < 0.10  # beyond the last data point
    assert out["losses"][-1, 1] < 0.01 * out["losses"][0, 1]  # the physics residual was driven down


def test_inverse_recovers_damping(inverse_run):
    out, _, _ = inverse_run
    assert out["mu"][0] < 1.5  # started from the wrong guess μ = 1
    assert abs(out["mu_final"] - MU) / MU < 0.10  # the ODE pins down the damping coefficient


def test_plain_network_fails_to_extrapolate(inverse_run, plain_run):
    out, t_plot, t_d = inverse_run
    plain, _, _ = plain_run
    exact = equation.exact_solution(t_plot, M, MU, K)
    data_region, extrap = t_plot <= t_d.max(), t_plot > t_d.max()
    x_plain = plain["snapshots"][-1]
    assert _rel_err(x_plain[data_region], exact[data_region]) < 0.1  # fits the data fine...
    err_plain = _rel_err(x_plain[extrap], exact[extrap])
    err_pinn = _rel_err(out["snapshots"][-1][extrap], exact[extrap])
    assert err_plain > 0.5 and err_plain > 5 * err_pinn  # ...but without physics it cannot extrapolate
    assert torch.all(plain["losses"][:, 1] == 0)  # no physics term at all


def test_reproducible_with_seed():
    a, _, _ = _train(1e-4, True, 50)
    b, _, _ = _train(1e-4, True, 50)
    assert torch.equal(a["snapshots"], b["snapshots"])
    assert a["mu_final"] == b["mu_final"]
