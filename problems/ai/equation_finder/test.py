"""Checks, written first (TDD): the derivative engine and the weak form are exact where calculus says they must be,
the simulated data obey their laws (closed-form heat kernel, Cole–Hopf Burgers, a Yee FDTD that keeps ∇·B = 0 and
conserves energy), STRidge recovers a planted sparse vector exactly, every target is recovered with the right terms
and coefficients from clean data, and the written LaTeX equation, read back alone, reproduces the fitted model.

Run with `pytest problems/ai/equation_finder/test.py` (CPU, seeded, float64).
"""

import math

import numpy as np
import pytest
import torch

from . import equation as eq
from . import solve
from . import tensors as T

DT = torch.float64


def _grid_2d(nx=128, nt=64, L=2 * math.pi, t_end=1.0):
    x = torch.arange(nx, dtype=DT) * L / nx
    t = torch.linspace(0, t_end, nt, dtype=DT)
    return x, t, x[:, None], t[None, :]


# ---------------------------------------------------------------- the derivative engine
def test_fd_kernels_are_the_textbook_central_differences():
    assert np.allclose(eq.fd_kernel(1), [-0.5, 0, 0.5])
    assert np.allclose(eq.fd_kernel(2), [1, -2, 1])
    assert np.allclose(eq.fd_kernel(3), [-0.5, 1, 0, -1, 0.5])


def test_savitzky_golay_kernels_are_exact_on_polynomials():
    j = np.arange(-5, 6, dtype=float)
    f = 0.3 - 1.2 * j + 0.7 * j**2 - 0.05 * j**3 + 0.01 * j**4  # degree 4, value/derivatives at 0 known
    exact = [0.3, -1.2, 1.4, -0.3]
    for d in range(4):
        assert abs(np.dot(eq.sg_kernel(5, 4, d), f) - exact[d]) < 1e-10
    assert abs(np.sum(eq.sg_kernel(5, 3, 0)) - 1) < 1e-12  # smoothing keeps constants


def test_library_terms_equal_analytic_derivatives():
    """u = sin(x) e^{-t}: every column of the strong-form library equals its analytic value (2nd-order FD error)."""
    x, t, X, Tt = _grid_2d()
    u = torch.sin(X) * torch.exp(-Tt)
    th, ut, crop = eq.strong_library(u[None], (x[1] - x[0], t[1] - t[0]), eq.SCALAR_1D, "fd")
    Xc, Tc = X.expand_as(u)[crop].reshape(-1), Tt.expand_as(u)[crop].reshape(-1)
    s, c, e = torch.sin(Xc), torch.cos(Xc), torch.exp(-Tc)
    analytic = {"u": s * e, "u^2": (s * e) ** 2, "u_{x}": c * e, "u u_{x}": s * c * e**2, "u_{xx}": -s * e,
                "u_{xxx}": -c * e, "\\partial_{xx}(u^2)": 2 * e**2 * (c**2 - s**2)}  # fmt: skip
    names = [tm.latex for tm in eq.SCALAR_1D]
    for k, v in analytic.items():
        assert torch.allclose(th[:, names.index(k)], v, atol=3e-3), k
    assert torch.allclose(ut[:, 0], -s * e, atol=3e-3)  # ∂_t u (central difference in time)


def test_weak_form_moves_the_derivatives_onto_the_test_function():
    """∫ φ D^α f(u) computed by integrating by parts (weak column) equals the integral of the analytic strong term,
    up to the rectangle-rule quadrature error (smooth bumps, so it is tiny)."""
    x, t, X, Tt = _grid_2d(256, 128)
    u = torch.sin(X) * torch.exp(-Tt)
    h = (float(x[1] - x[0]), float(t[1] - t[0]))
    phi = eq.test_functions(u.shape, h, radius=(20, 16), stride=(16, 12))
    th, ut = eq.weak_library(u[None], h, eq.SCALAR_1D, phi)
    s, c, e = torch.sin(X), torch.cos(X), torch.exp(-Tt)
    strong = {"u_{xx}": -s * e, "u u_{x}": s * c * e**2, "u_{xxx}": -c * e}
    names = [tm.latex for tm in eq.SCALAR_1D]
    for k, v in strong.items():
        direct = eq.contract(v, [phi[0][0], phi[1][0]]).reshape(-1) * h[0] * h[1]
        assert (th[:, names.index(k)] - direct).abs().max() < 1e-4 * direct.abs().max(), k  # quadrature error only
    direct_ut = eq.contract(-s * e, [phi[0][0], phi[1][0]]).reshape(-1) * h[0] * h[1]
    assert (ut[:, 0] - direct_ut).abs().max() < 1e-6 * direct_ut.abs().max()


# ---------------------------------------------------------------- sparse regression
def test_stridge_recovers_a_planted_sparse_vector_exactly():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(400, 15, generator=g, dtype=DT)
    xi = torch.zeros(2, 15, dtype=DT)
    xi[0, [1, 4, 9]] = torch.tensor([2.0, -0.7, 0.05], dtype=DT)
    xi[1, [0, 14]] = torch.tensor([-3.0, 1.5], dtype=DT)
    b = A @ xi.T
    found = solve.stridge(A, b, tol=0.01)
    assert torch.equal(found != 0, xi != 0)
    assert torch.allclose(found, xi, atol=1e-10)


def test_bootstrap_spread_matches_the_least_squares_standard_error():
    g = torch.Generator().manual_seed(2)
    n, sig = 2000, 0.3
    A = torch.randn(n, 2, generator=g, dtype=DT)
    b = A @ torch.tensor([1.0, -2.0], dtype=DT) + sig * torch.randn(n, generator=g, dtype=DT)
    est, err = solve.bootstrap(A, b, n_boot=400, seed=0)
    se = sig * torch.linalg.inv(A.T @ A).diagonal().sqrt()  # textbook OLS standard error
    assert torch.allclose(est, torch.tensor([1.0, -2.0], dtype=DT), atol=4 * float(se.max()))
    assert torch.all((err / se - 1).abs() < 0.25)


# ---------------------------------------------------------------- the simulated data obey their laws
def test_heat_data_is_the_heat_kernel():
    """One Gaussian bump spreads as a Gaussian with σ² = σ0² + 2Dt; on the periodic rod, add its periodic images."""
    ds = T.heat(bumps=[(5.0, 0.4, 1.0)])
    x, t, D, L = ds["coords"][0], ds["coords"][-1], T.HEAT_D, 10.0
    s2 = 0.4**2 + 2 * D * t[None, :]
    exact = sum(0.4 / s2.sqrt() * torch.exp(-((x[:, None] - 5.0 - m * L) ** 2) / (2 * s2)) for m in range(-4, 5))
    assert torch.allclose(ds["U"][0], exact, atol=1e-10)


def test_burgers_cole_hopf_data_solve_burgers():
    ds = T.burgers()
    u = ds["U"][0]
    th, ut, _ = eq.strong_library(u[None], ds["h"], eq.SCALAR_1D, "fd")
    names = [tm.latex for tm in eq.SCALAR_1D]
    resid = ut[:, 0] + th[:, names.index("u u_{x}")] - T.BURGERS_NU * th[:, names.index("u_{xx}")]
    assert resid.norm() / ut[:, 0].norm() < 5e-3  # FD truncation only; exact Cole–Hopf data


def test_lorenz_trajectories_are_converged():
    a = T.lorenz(n_traj=2, t_end=1.0, substeps=2)["U"]
    b = T.lorenz(n_traj=2, t_end=1.0, substeps=8)["U"]
    assert (a - b).abs().max() < 1e-5


def test_fdtd_keeps_div_b_zero_and_conserves_energy():
    sim = T.yee_fdtd(n=64, steps=200, record=True)
    assert sim["div_b"].abs().max() < 1e-10 * sim["b_scale"]  # Yee: discrete ∇·B = 0 to round-off
    energy = sim["energy"]
    assert (energy.max() - energy.min()) / energy.mean() < 2e-3


def test_fdtd_courant_limit_is_refused():
    with pytest.raises(ValueError):
        T.yee_fdtd(n=16, steps=2, courant=0.8)


# ---------------------------------------------------------------- recovery from clean data
@pytest.fixture(scope="module")
def clean():
    return {name: solve.discover(T.dataset(name, small=True), "fd", 0.0, seed=0, n_boot=50) for name in T.TARGETS}


def test_lorenz_recovered(clean):
    r = clean["lorenz"]
    assert r["success"]
    names = [tm.latex for tm in r["terms"]]
    xi = r["xi"]
    sigma = (xi[0, names.index("y")] - xi[0, names.index("x")]) / 2
    assert abs(sigma / 10 - 1) < 0.01
    assert abs(xi[1, names.index("x")] / 28 - 1) < 0.01 and abs(xi[1, names.index("x z")] + 1) < 0.01
    assert abs(xi[2, names.index("z")] / (-8 / 3) - 1) < 0.01 and abs(xi[2, names.index("x y")] - 1) < 0.01


def test_heat_recovered(clean):
    r = clean["heat"]
    assert r["success"]
    D = r["xi"][0, [tm.latex for tm in r["terms"]].index("u_{xx}")]
    assert abs(D / T.HEAT_D - 1) < 0.01


def test_burgers_recovered(clean):
    r = clean["burgers"]
    names = [tm.latex for tm in r["terms"]]
    assert r["success"]
    assert abs(r["xi"][0, names.index("u u_{x}")] + 1) < 0.02
    assert abs(r["xi"][0, names.index("u_{xx}")] / T.BURGERS_NU - 1) < 0.02


def test_maxwell_recovered(clean):
    r = clean["maxwell"]
    names = [tm.latex for tm in r["terms"]]
    assert r["success"]
    c2 = (r["xi"][0, names.index("\\partial_x B_y")] - r["xi"][0, names.index("\\partial_y B_x")]) / 2
    assert abs(math.sqrt(c2) / T.C_LIGHT - 1) < 0.01
    assert abs(r["xi"][1, names.index("\\partial_y E_z")] + 1) < 0.01
    assert abs(r["xi"][2, names.index("\\partial_x E_z")] - 1) < 0.01


def test_divergence_free_constraint_is_the_library_null_vector():
    """The smallest singular vector of the Maxwell library is ∂_x B_x + ∂_y B_y: the data find ∇·B = 0 by themselves."""
    ds = T.dataset("maxwell", small=True)
    th, _, _ = eq.strong_library(ds["U"], ds["h"], ds["terms"], "fd")
    vec, sv = solve.null_vector(th)
    names = [tm.latex for tm in ds["terms"]]
    i, j = names.index("\\partial_x B_x"), names.index("\\partial_y B_y")
    assert abs(abs(vec[i]) - 1 / math.sqrt(2)) < 0.01 and abs(vec[i] - vec[j]) < 0.01
    assert sv[0] < 0.05 * sv[1]  # a clear gap: the next singular value is far larger


def test_annealed_gate_relaxation_finds_the_heat_equation():
    ds = T.dataset("heat", small=True)
    th, ut, _ = eq.strong_library(ds["U"], ds["h"], ds["terms"], "fd")
    out = solve.gate_search(th, ut, steps=600, seed=0)
    assert torch.equal(out["support"], ds["truth"] != 0)


# ---------------------------------------------------------------- the written equation
def test_written_equation_reads_back_and_reproduces_the_model(clean):
    for name, r in clean.items():
        ds = T.dataset(name, small=True)
        th, _, _ = eq.strong_library(ds["U"], ds["h"], ds["terms"], "fd")
        for e, lhs in enumerate(ds["lhs"]):
            line = solve.write_equation(lhs, r["terms"], r["xi"][e], digits=12)
            xi_back = solve.read_equation(line, r["terms"])
            assert torch.allclose(xi_back, r["xi"][e], rtol=1e-10, atol=0)
            scale = float((th.abs() @ r["xi"][e].abs()).max())  # size of the largest single term
            assert (th @ xi_back - th @ r["xi"][e]).abs().max() <= 1e-10 * scale


def test_equation_text_of_a_known_model():
    terms = eq.SCALAR_1D
    xi = torch.zeros(len(terms), dtype=DT)
    names = [tm.latex for tm in terms]
    xi[names.index("u u_{x}")], xi[names.index("u_{xx}")] = -1.0, 0.1
    err = torch.zeros_like(xi)
    err[names.index("u u_{x}")], err[names.index("u_{xx}")] = 0.0021, 0.00034
    assert (
        solve.write_equation("u_t", terms, xi, err=err)
        == "u_t = -(1.0000 \\pm 0.0021)\\, u u_{x} + (0.10000 \\pm 0.00034)\\, u_{xx}"
    )
    assert solve.write_equation("u_t", terms, xi, digits=3) == "u_t = -1\\, u u_{x} + 0.1\\, u_{xx}"
    assert torch.equal(solve.read_equation(solve.write_equation("u_t", terms, xi, err=err), terms), xi)


# ---------------------------------------------------------------- the noise study (slow)
@pytest.mark.slow
def test_noise_study_weak_form_survives_noise_that_breaks_finite_differences():
    rows = solve.noise_study(levels=(0.0, 0.01, 0.05), methods=("fd", "weak"), small=True, n_boot=20)
    ok = {(r["target"], r["method"], r["noise"]): r["success"] for r in rows}
    assert all(ok[t, m, 0.0] for t in T.TARGETS for m in ("fd", "weak"))
    assert all(ok[t, "weak", 0.01] for t in T.TARGETS)
    assert sum(ok[t, "fd", 0.05] for t in T.TARGETS) < sum(ok[t, "weak", 0.05] for t in T.TARGETS)


# ---------------------------------------------------------------- the found equation as a simulator
def test_true_equation_resimulated_reproduces_the_data():
    """Integrating the true law from the first snapshot gives back the data (heat: exact; Burgers: Cole–Hopf)."""
    for name, tol in (("heat", 1e-8), ("burgers", 1e-6)):
        ds = T.dataset(name)
        sim = solve.simulate_found(ds, ds["truth"], n_times=40)
        data = ds["U"][..., :40]
        assert (sim - data).norm() / data.norm() < tol, name
