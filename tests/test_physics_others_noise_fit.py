"""Physics checks for others/langevin_noise and others/measurement_fitting."""

import math
import re

import torch

from problems.others.langevin_noise import equation as lv_eq
from problems.others.langevin_noise import solve as lv_solve
from problems.others.langevin_noise import tensors as lv_t
from problems.others.measurement_fitting import equation as mf_eq
from problems.others.measurement_fitting import solve as mf_solve
from problems.others.measurement_fitting import tensors as mf_t

CPU = torch.device("cpu")


# --- Langevin equation -------------------------------------------------------------------------------


def _langevin(T=1.0, gamma=0.5, k=1.0, m=1.0, B=2000, t_max=40.0, dt=None, seed=0, x0=None):
    x, v = lv_t.initial_state(B, CPU)
    if x0 is not None:
        x = torch.full_like(x, x0)
    dt = dt or lv_t.time_step(gamma, k, m)
    return lv_solve.solve(x, v, gamma, k, m, T, t_max, dt, lv_t.generator(CPU, seed), n_saved=400)


def test_langevin_equipartition():
    T, k, m = 1.5, 2.0, 0.5
    times, x, v = _langevin(T=T, gamma=1.0, k=k, m=m, B=4000, t_max=30.0, dt=0.002)
    late = times > 10.0
    x2 = lv_eq.ensemble_mean_square(x)[late].mean().item()
    v2 = lv_eq.ensemble_mean_square(v)[late].mean().item()
    x2_eq, v2_eq = lv_eq.equipartition(T, k, m)
    assert abs(x2 / x2_eq - 1) < 0.04, (x2, x2_eq)
    assert abs(v2 / v2_eq - 1) < 0.04, (v2, v2_eq)
    # the final positions are Gaussian with variance k_B T / k: kurtosis ⟨x⁴⟩/⟨x²⟩² ≈ 3
    xe = x[-1]
    assert abs((xe**4).mean() / (xe**2).mean() ** 2 - 3) < 0.25


def test_langevin_zero_temperature_is_damped_oscillator():
    gamma, k, m = 0.5, 1.0, 1.0
    times, x, v = _langevin(T=0.0, gamma=gamma, k=k, m=m, B=3, t_max=20.0, dt=0.001)
    exact = lv_eq.damped_oscillator(times, lv_t.X0, gamma, k, m)
    assert torch.allclose(x[:, 0], x[:, 2])  # no noise: every particle follows the same path
    assert (x[:, 0] - exact).abs().max() < 5e-3
    # the overdamped branch of the analytic formula solves the ODE too (finite differences)
    t = torch.linspace(0, 5, 5001, dtype=torch.float64)
    xo = lv_eq.damped_oscillator(t, 1.0, 5.0, 1.0, 1.0)
    h = (t[1] - t[0]).item()
    acc = (xo[2:] - 2 * xo[1:-1] + xo[:-2]) / h**2
    vel = (xo[2:] - xo[:-2]) / (2 * h)
    assert (acc + 5.0 * vel + xo[1:-1]).abs().max() < 1e-5


def test_langevin_noise_increment_scales_with_sqrt_dt():
    gamma, T, m = 0.7, 2.0, 1.3
    for dt in (1e-2, 1e-4):
        amp = lv_eq.noise_amplitude(gamma, T, m, dt)
        assert math.isclose(amp**2, 2 * gamma * T * dt / m**2, rel_tol=1e-12)
        # one step from rest at x = 0: the velocity change is pure noise with variance 2γk_BTΔt/m²
        x = torch.zeros(200_000, dtype=torch.float64)
        xi = torch.randn(x.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        _, dv = lv_eq.euler_maruyama_step(x, x.clone(), xi, gamma, 1.0, m, T, dt)
        assert abs(dv.var().item() / (2 * gamma * T * dt / m**2) - 1) < 0.02
    # halving Δt shrinks the kick by √2, not by 2
    ratio = lv_eq.noise_amplitude(gamma, T, m, 0.01) / lv_eq.noise_amplitude(gamma, T, m, 0.005)
    assert math.isclose(ratio, math.sqrt(2), rel_tol=1e-12)


def test_langevin_overdamped_variance_relaxation():
    T, gamma, k, m = 1.0, 5.0, 1.0, 0.05  # m/γ = 0.01 ≪ γ/k = 5: overdamped Brownian motion
    times, x, _ = _langevin(T=T, gamma=gamma, k=k, m=m, B=4000, t_max=6.0, dt=0.001, x0=0.0)
    x2 = lv_eq.ensemble_mean_square(x)
    expected = T / k * (1 - torch.exp(-2 * k / gamma * times))  # ⟨x²⟩ = (k_BT/k)(1 - e^{-2kt/γ})
    assert (x2 - expected)[times > 0.5].abs().max() < 0.08
    # rate from the half-way point: e^{-2kt/γ} = 1/2 at t = γ ln2 / 2k
    i = int((x2 > 0.5 * T / k).int().argmax())
    rate = math.log(2) / times[i].item()
    assert abs(rate / (2 * k / gamma) - 1) < 0.15


def test_langevin_reproducible_with_seed():
    _, xa, _ = _langevin(B=50, t_max=5.0, seed=3)
    _, xb, _ = _langevin(B=50, t_max=5.0, seed=3)
    _, xc, _ = _langevin(B=50, t_max=5.0, seed=4)
    assert torch.equal(xa, xb)
    assert not torch.equal(xa, xc)


# --- measurement fitting -------------------------------------------------------------------------------


def _truth() -> dict:
    """The true parameters stated in the CSV header."""
    head = "".join(line for line in mf_t.DATA.read_text(encoding="utf-8").splitlines(True) if line.startswith("#"))
    get = lambda name: float(re.search(rf"\b{name} = ([-\d.]+)", head).group(1))  # noqa: E731
    return {n: get(n) for n in ("A", "gamma", "omega", "phi", "offset", "s")}


def _fit(jitter=True):
    t, x, sigma = mf_t.load_measurements(CPU)
    p0 = mf_t.initial_parameters(t, x, sigma, 4.6, 0.5)
    p, history, losses = mf_solve.fit(p0, t, x, sigma, 0.03, 300, jitter)
    return t, x, sigma, p, history, losses


def test_measurements_load():
    t, x, sigma = mf_t.load_measurements(CPU)
    assert t.shape == x.shape == sigma.shape == (80,)
    assert t.dtype == torch.float64
    assert (sigma > 0).all() and (t[1:] > t[:-1]).all()


def test_fit_recovers_true_parameters_and_jitter():
    t, x, sigma, p, _, losses = _fit()
    assert losses[-1] < losses[0] / 5
    C = mf_eq.covariance_from_hessian(p, t, x, sigma)
    err = C.diagonal().sqrt()
    true = _truth()
    for i, name in enumerate(["A", "gamma", "omega", "phi", "offset"]):
        assert abs(p[i].item() - true[name]) < 3 * err[i].item(), name
    s, s_err = p[5].exp().item(), p[5].exp().item() * err[5].item()
    assert abs(s - true["s"]) < 2 * s_err
    assert s > 3 * s_err  # the fit clearly finds extra scatter


def test_fit_chi2_per_dof_and_normalised_residuals():
    t, x, sigma, p, _, _ = _fit()
    r = mf_eq.residuals(p, t, x, sigma)
    assert 0.6 < (r @ r).item() / (len(t) - 6) < 1.5
    assert abs(r.std().item() - 1) < 0.15
    # without the error term s the same data look badly over-scattered
    _, _, _, q, _, _ = _fit(jitter=False)
    assert mf_eq.chi2(q, t, x, sigma).item() / (len(t) - 5) > 1.8


def test_zero_jitter_is_weighted_least_squares():
    t, x, sigma = mf_t.load_measurements(CPU)
    _, _, _, q, _, _ = _fit(jitter=False)
    # the s = 0 likelihood differs from χ² by a constant, so their gradients agree everywhere ...
    p = q.clone().requires_grad_(True)
    chi = mf_eq.chi2(p, t, x, sigma)
    (g_chi,) = torch.autograd.grad(chi, p)
    tiny = torch.cat([q[:5], torch.tensor([-60.0], dtype=q.dtype)]).requires_grad_(True)  # s = e^{-60} ≈ 0
    nll = mf_eq.neg2_log_likelihood(tiny, t, x, sigma)
    (g_nll,) = torch.autograd.grad(nll, tiny)
    assert torch.allclose(g_chi[:5], g_nll[:5], atol=1e-6)
    assert math.isclose(nll.item() - chi.item(), torch.log(2 * math.pi * sigma**2).sum().item(), rel_tol=1e-9)
    # ... and the χ² minimum is stationary: the gradient vanishes there
    assert g_chi[:5].abs().max() < 1e-4
    # it matches the Gauss–Newton step: (JᵀJ) δ = -Jᵀ r ≈ 0 at the optimum
    J = mf_eq.jacobian(q, t, x, sigma, jitter=False)
    r = mf_eq.residuals(q, t, x, sigma, jitter=False)
    delta = torch.linalg.solve(J.T @ J, -J.T @ r)
    assert delta.abs().max() < 1e-6


def test_lstsq_matches_normal_equations():
    t, x, sigma = mf_t.load_measurements(CPU)
    for gamma, omega in [(0.25, 5.0), (0.5, 4.6)]:
        a = mf_eq.linear_fit(t, x, sigma, gamma, omega)
        b = mf_eq.normal_equations(t, x, sigma, gamma, omega)
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-8)
    # at the nonlinear optimum the linear sub-problem returns the same A, φ, c
    _, _, _, p, _, _ = _fit()
    w = mf_eq.total_sigma(sigma, p[5])
    A, phi = mf_eq.amplitude_phase(mf_eq.linear_fit(t, x, w, p[1].item(), p[2].item()))
    assert abs(A.item() - p[0].item()) < 1e-5 and abs(phi.item() - p[3].item()) < 1e-6


def test_jacobian_and_covariance_match_finite_differences():
    t, x, sigma, p, _, _ = _fit()
    J = mf_eq.jacobian(p, t, x, sigma)
    h = 1e-6
    J_fd = torch.stack(
        [
            (mf_eq.residuals(p + h * e, t, x, sigma) - mf_eq.residuals(p - h * e, t, x, sigma)) / (2 * h)
            for e in torch.eye(6, dtype=p.dtype)[:5]
        ],
        dim=1,
    )
    assert torch.allclose(J, J_fd, rtol=1e-5, atol=1e-7)
    C_gn = mf_eq.covariance_from_jacobian(J)
    C_fd = torch.linalg.inv(torch.einsum("ip,iq->pq", J_fd, J_fd))
    assert torch.allclose(C_gn, C_fd, rtol=1e-4)
    # the Hessian covariance (which includes s) agrees with Gauss–Newton on the θ block to ~10 %
    C_h = mf_eq.covariance_from_hessian(p, t, x, sigma)
    assert torch.allclose(C_h.diagonal()[:5].sqrt(), C_gn.diagonal().sqrt(), rtol=0.1)
    assert C_h.shape == (6, 6) and (torch.linalg.eigvalsh(C_h) > 0).all()
