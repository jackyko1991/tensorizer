"""Checks written first (TDD): what any correct self-similar blow-up search must satisfy.

Run with `pytest problems/fluids/blowup_profile_search/test.py`. Burgers' equation is the fully checkable warm-up: the
smooth profiles are known in closed (implicit) form, y = −U − U^{2i+3} with λ = 1/(2i+2), so the search must find
exactly those exponents and profiles, and the self-similar field built from a profile must solve the PDE. For the
Córdoba–Córdoba–Fontelos (CCF) equation the Hilbert transform is checked on known pairs, and training must lower the
residual.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_exact_burgers_profiles_solve_the_profile_equation():
    y = torch.linspace(-20, 20, 801, dtype=torch.float64)
    for i in range(3):  # λ = 1/2, 1/4, 1/6
        lam = 1 / (2 * i + 2)
        U, dU = equation.burgers_exact(y, i)
        assert torch.allclose(-U - U ** (2 * i + 3), y, atol=1e-12)  # the implicit form itself
        R = equation.burgers_residual(y, U, dU, lam)
        assert R.abs().max() < 1e-12  # −λU + ((1+λ)y + U)U′ = 0
        assert abs(dU[400] + 1) < 1e-12 and abs(U[400]) < 1e-14  # normalisation U(0) = 0, U′(0) = −1
        assert torch.allclose(U, -U.flip(0), atol=1e-13)  # odd


def test_self_similar_field_solves_inviscid_burgers():
    """u(x, t) = (T − t)^λ U(x / (T − t)^{1+λ}) with the exact profile, plugged into u_t + u u_x = 0 by finite
    differences in t and x."""
    x = torch.linspace(-1, 1, 201, dtype=torch.float64)
    T, t, h = 1.0, 0.6, 1e-5
    for i in range(2):
        lam = 1 / (2 * i + 2)

        def u(x, t, lam=lam, i=i):
            tau = T - t
            return tau**lam * equation.burgers_exact(x / tau ** (1 + lam), i)[0]

        u_t = (u(x, t + h) - u(x, t - h)) / (2 * h)
        u_x = (u(x + h, t) - u(x - h, t)) / (2 * h)
        scale = u(x, t).abs().max() / (T - t)
        assert (u_t + u(x, t) * u_x).abs().max() < 1e-6 * scale


def test_gradient_blows_up_like_one_over_time_to_go():
    tau = torch.tensor([1e-1, 1e-2, 1e-3], dtype=torch.float64)
    for lam in (0.5, 0.25):
        slope = equation.max_slope(tau, lam)  # max |u_x| = |U′(0)| / (T − t)
        assert torch.allclose(slope * tau, torch.ones_like(tau))


def test_spectral_representation_holds_the_exact_profile():
    """The far-field-aware spectral basis can represent y = −U − U³ to near round-off."""
    r = tensors.chebyshev_points(120, CPU)
    for i in range(2):
        lam = 1 / (2 * i + 2)
        c = solve.project_profile(lambda y, i=i: equation.burgers_exact(y, i)[0], lam, r, n_modes=48)
        y = torch.linspace(-8, 8, 161, dtype=torch.float64)
        U, _ = equation.burgers_profile(c, torch.tensor(lam, dtype=torch.float64), y)
        assert (U - equation.burgers_exact(y, i)[0]).abs().max() < 1e-8


def test_funnel_only_at_the_quantised_exponents():
    """Residual after fitting U at fixed λ: tiny at λ = 1/2 and 1/4, large in between (non-smooth profiles)."""
    lams = torch.tensor([0.25, 0.37, 0.5, 0.6], dtype=torch.float64)
    res = solve.burgers_funnel(lams, level=0, iters=40)
    assert res[2] < 1e-10 and res[0] < 1e-6
    assert res[1] > 1e3 * res[2] and res[3] > 1e3 * res[2]
    assert res[1] > 30 * res[0]


@pytest.mark.slow
def test_training_recovers_stable_and_unstable_exponents():
    out = solve.burgers_search(torch.tensor([0.48, 0.51], dtype=torch.float64), level=0)
    assert (out["lam"] - 0.5).abs().max() < 1e-3  # stable λ = 1/2
    out1 = solve.burgers_search(torch.tensor([0.25, 0.27], dtype=torch.float64), level=1)  # U‴(0) = 0 imposed
    assert (out1["lam"] - 0.25).abs().max() < 1e-2  # first unstable λ = 1/4
    y = torch.linspace(-5, 5, 201, dtype=torch.float64)
    for out_, i in ((out, 0), (out1, 1)):
        U, _ = equation.burgers_profile(out_["coef"][0], out_["lam"][0], y)
        assert (U - equation.burgers_exact(y, i)[0]).abs().max() < 1e-4  # sup-norm on the window |y| ≤ 5


def test_hilbert_transform_of_known_functions():
    n = 256
    x = torch.arange(n, dtype=torch.float64) * (2 * math.pi / n)
    assert torch.allclose(equation.hilbert_periodic(torch.cos(3 * x)), torch.sin(3 * x), atol=1e-12)  # H[cos] = sin
    phi, y = tensors.circle_grid(n, CPU)  # y = tan(φ/2) maps the real line onto the circle
    f = 1 / (1 + y**2)
    assert torch.allclose(equation.hilbert_line(f), y / (1 + y**2), atol=1e-12)  # H[1/(1+x²)] = x/(1+x²)
    assert torch.allclose(equation.hilbert_line(y / (1 + y**2)), -f, atol=1e-12)  # H² = −1


def test_ccf_basis_hilbert_pairs_match_the_fft_transform():
    """The analytic Hilbert transform of the CCF basis (boundary values of (1 − iy)^{-s}, analytic in the upper half
    plane) agrees with the FFT transform on the mapped circle for smooth members (integer s)."""
    phi, y = tensors.circle_grid(512, CPU)
    for s in (1.0, 2.0, 3.0):
        omega, h_omega = equation.ccf_basis_pair(y, torch.tensor([s], dtype=torch.float64))
        assert torch.allclose(equation.hilbert_line(omega[:, 0]), h_omega[:, 0], atol=1e-10)


@pytest.mark.slow
def test_ccf_residual_decreases_during_training():
    out = solve.ccf_search(torch.tensor([1.2], dtype=torch.float64), n_golden=4, n_fit=10, polish=2, warm=25)
    hist = out["residual_history"][:, 0]
    assert hist[-1] < 0.01 * hist[0]  # from the initial guess to the trained profile
    assert torch.all(out["omega"][:, 0][1:] <= 1e-12)  # Ω < 0 for y > 0: one-signed odd profile (stable branch)


def test_closed_form_jacobians_agree_with_autograd():
    """Gauss–Newton uses the exact Jacobians of the (quadratic) residuals; autograd must give the same matrices."""
    torch.manual_seed(0)
    lam = torch.tensor([0.4, 0.55], dtype=torch.float64)
    r = tensors.chebyshev_points(30, CPU)
    C = 0.01 * torch.randn(2, 12, dtype=torch.float64)
    for level in (0, 2):
        _, J = equation.burgers_batch(C, lam, r, level)
        J_auto = torch.func.jacrev(lambda c, lv=level: equation.burgers_batch(c, lam, r, lv)[0])(C)  # (B, M, B, K)
        assert torch.allclose(J, torch.einsum("bmbk->bmk", J_auto), atol=1e-10)
    y = tensors.ccf_points(40, CPU)
    A = tensors.ccf_initial_coefficients(lam + 0.7, 2, 4) + 0.01 * torch.randn(2, 14, dtype=torch.float64)
    _, J = equation.ccf_batch(A, lam + 0.7, y, 2, 4)
    J_auto = torch.func.jacrev(lambda a: equation.ccf_batch(a, lam + 0.7, y, 2, 4)[0])(A)
    assert torch.allclose(J, torch.einsum("bmbk->bmk", J_auto), atol=1e-10)
