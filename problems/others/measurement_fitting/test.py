"""Physics checks, written first (TDD): what any correct measurement-fitting code must satisfy.

Run with `pytest problems/others/measurement_fitting/test.py`. Each test is a law of statistics, not an
implementation detail: the fit recovers the true parameters within their errors, χ²/dof ≈ 1 with unit normalised
residuals, the s = 0 likelihood is weighted least squares, and the covariance is the inverse Fisher matrix.
"""

import math
import re

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _truth() -> dict:
    """The true parameters stated in the CSV header."""
    head = "".join(line for line in tensors.DATA.read_text(encoding="utf-8").splitlines(True) if line.startswith("#"))
    get = lambda name: float(re.search(rf"\b{name} = ([-\d.]+)", head).group(1))  # noqa: E731
    return {n: get(n) for n in ("A", "gamma", "omega", "phi", "offset", "s")}


def _fit(jitter=True):
    t, x, sigma = tensors.load_measurements(CPU)
    p0 = tensors.initial_parameters(t, x, sigma, 4.6, 0.5)
    p, history, losses = solve.fit(p0, t, x, sigma, 0.03, 300, jitter)
    return t, x, sigma, p, history, losses


def test_measurements_load():
    t, x, sigma = tensors.load_measurements(CPU)
    assert t.shape == x.shape == sigma.shape == (80,)
    assert t.dtype == torch.float64
    assert (sigma > 0).all() and (t[1:] > t[:-1]).all()  # errors are positive, time runs forward


def test_fit_recovers_true_parameters_and_jitter():
    t, x, sigma, p, _, losses = _fit()
    assert losses[-1] < losses[0] / 5
    C = equation.covariance_from_hessian(p, t, x, sigma)
    err = C.diagonal().sqrt()
    true = _truth()
    for i, name in enumerate(["A", "gamma", "omega", "phi", "offset"]):
        assert abs(p[i].item() - true[name]) < 3 * err[i].item(), name  # unbiased: truth within 3σ
    s, s_err = p[5].exp().item(), p[5].exp().item() * err[5].item()
    assert abs(s - true["s"]) < 2 * s_err
    assert s > 3 * s_err  # the fit clearly finds extra scatter


def test_fit_chi2_per_dof_and_normalised_residuals():
    t, x, sigma, p, _, _ = _fit()
    r = equation.residuals(p, t, x, sigma)
    assert 0.6 < (r @ r).item() / (len(t) - 6) < 1.5  # a good fit: χ²/dof ≈ 1
    assert abs(r.std().item() - 1) < 0.15  # pulls are unit Gaussian
    # without the error term s the same data look badly over-scattered
    _, _, _, q, _, _ = _fit(jitter=False)
    assert equation.chi2(q, t, x, sigma).item() / (len(t) - 5) > 1.8


def test_zero_jitter_is_weighted_least_squares():
    t, x, sigma = tensors.load_measurements(CPU)
    _, _, _, q, _, _ = _fit(jitter=False)
    # the s = 0 likelihood differs from χ² by a constant, so their gradients agree everywhere ...
    p = q.clone().requires_grad_(True)
    chi = equation.chi2(p, t, x, sigma)
    (g_chi,) = torch.autograd.grad(chi, p)
    tiny = torch.cat([q[:5], torch.tensor([-60.0], dtype=q.dtype)]).requires_grad_(True)  # s = e^{-60} ≈ 0
    nll = equation.neg2_log_likelihood(tiny, t, x, sigma)
    (g_nll,) = torch.autograd.grad(nll, tiny)
    assert torch.allclose(g_chi[:5], g_nll[:5], atol=1e-6)
    # Gaussian likelihood: −2 ln L = χ² + Σ ln 2πσ²
    assert math.isclose(nll.item() - chi.item(), torch.log(2 * math.pi * sigma**2).sum().item(), rel_tol=1e-9)
    # ... and the χ² minimum is stationary: the gradient vanishes there
    assert g_chi[:5].abs().max() < 1e-4
    # it matches the Gauss–Newton step: (JᵀJ) δ = -Jᵀ r ≈ 0 at the optimum
    J = equation.jacobian(q, t, x, sigma, jitter=False)
    r = equation.residuals(q, t, x, sigma, jitter=False)
    delta = torch.linalg.solve(J.T @ J, -J.T @ r)
    assert delta.abs().max() < 1e-6


def test_lstsq_matches_normal_equations():
    t, x, sigma = tensors.load_measurements(CPU)
    for gamma, omega in [(0.25, 5.0), (0.5, 4.6)]:
        a = equation.linear_fit(t, x, sigma, gamma, omega)
        b = equation.normal_equations(t, x, sigma, gamma, omega)
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-8)
    # at the nonlinear optimum the linear sub-problem returns the same A, φ, c
    _, _, _, p, _, _ = _fit()
    w = equation.total_sigma(sigma, p[5])
    A, phi = equation.amplitude_phase(equation.linear_fit(t, x, w, p[1].item(), p[2].item()))
    assert abs(A.item() - p[0].item()) < 1e-5 and abs(phi.item() - p[3].item()) < 1e-6


def test_jacobian_and_covariance_match_finite_differences():
    t, x, sigma, p, _, _ = _fit()
    J = equation.jacobian(p, t, x, sigma)
    h = 1e-6
    J_fd = torch.stack(
        [
            (equation.residuals(p + h * e, t, x, sigma) - equation.residuals(p - h * e, t, x, sigma)) / (2 * h)
            for e in torch.eye(6, dtype=p.dtype)[:5]
        ],
        dim=1,
    )
    assert torch.allclose(J, J_fd, rtol=1e-5, atol=1e-7)
    C_gn = equation.covariance_from_jacobian(J)
    C_fd = torch.linalg.inv(torch.einsum("ip,iq->pq", J_fd, J_fd))
    assert torch.allclose(C_gn, C_fd, rtol=1e-4)  # covariance = inverse Fisher information (JᵀJ)⁻¹
    # the Hessian covariance (which includes s) agrees with Gauss–Newton on the θ block to ~10 %
    C_h = equation.covariance_from_hessian(p, t, x, sigma)
    assert torch.allclose(C_h.diagonal()[:5].sqrt(), C_gn.diagonal().sqrt(), rtol=0.1)
    assert C_h.shape == (6, 6) and (torch.linalg.eigvalsh(C_h) > 0).all()  # a covariance is positive definite
