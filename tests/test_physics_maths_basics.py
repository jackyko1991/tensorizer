"""Checks for the beginner maths problems: Fourier series and Gibbs, quadrature orders, 2×2 linear maps."""

import math

import torch

from problems.maths.fourier_gibbs import equation as fourier
from problems.maths.fourier_gibbs import solve as fourier_solve
from problems.maths.fourier_gibbs import tensors as fourier_tensors
from problems.maths.linear_maps_2d import equation as maps
from problems.maths.linear_maps_2d import solve as maps_solve
from problems.maths.linear_maps_2d import tensors as maps_tensors
from problems.maths.numerical_integration import equation as quad
from problems.maths.numerical_integration import solve as quad_solve
from problems.maths.numerical_integration import tensors as quad_tensors

CPU = torch.device("cpu")
GEN = torch.Generator().manual_seed(0)


def fourier_setup(n_x: int = 8192, n_max: int = 101):
    x, w = fourier_tensors.grid(n_x, CPU)
    f = fourier_tensors.waves(x)
    b = fourier_tensors.basis(x, n_max)
    return x, w, f, b, fourier.coefficients(f, b, w)


# --- Fourier series ------------------------------------------------------------------------------------------------


def test_square_wave_coefficients_are_4_over_pi_n_for_odd_n():
    _, _, _, _, c = fourier_setup()
    n = torch.arange(1, c.shape[1] + 1, dtype=torch.float64)
    expected = torch.where(n % 2 == 1, 4 / (math.pi * n), torch.zeros_like(n))
    assert torch.allclose(c[0], expected, atol=1e-5)
    assert torch.allclose(c, fourier_tensors.exact_coefficients(c.shape[1], CPU), atol=1e-5)


def test_gibbs_overshoot_tends_to_8_95_percent_of_the_jump():
    _, _, f, b, c = fourier_setup()
    s = torch.stack([fourier.partial_sum(b, c, n) for n in (25, 51, 101)])
    over = fourier_solve.overshoot(s, f, fourier_tensors.JUMP)
    gibbs = 0.08949  # (1/π) Si(π) − 1/2
    assert torch.allclose(over[:, :2], torch.full_like(over[:, :2], gibbs), atol=2e-3)  # square and sawtooth
    assert over[-1, 2] < 1e-3  # the continuous triangle wave has no overshoot


def test_parseval_energy_equals_sum_of_squared_coefficients():
    _, w, f, _, _ = fourier_setup()
    exact = fourier_tensors.exact_coefficients(200000, CPU)
    # (1/π)∫ f² dx: 2 for the square wave, 2/3 for the sawtooth and the triangle wave
    assert torch.allclose((exact**2).sum(dim=1), torch.tensor([2.0, 2 / 3, 2 / 3], dtype=torch.float64), atol=1e-5)
    assert torch.allclose(fourier.energy(f, w), torch.tensor([2.0, 2 / 3, 2 / 3], dtype=torch.float64), atol=1e-6)
    # the squared L² error of a partial sum is the energy of the missing coefficients
    _, w, f, b, c = fourier_setup(n_max=60)
    err = fourier.l2_error_sq(f, fourier.partial_sum(b, c, 20), w)
    assert torch.allclose(err, fourier.energy(f, w) - (c[:, :20] ** 2).sum(dim=1), atol=1e-9)


def test_l2_error_decays_like_1_over_n_for_jumps_and_1_over_n_cubed_for_a_kink():
    _, w, f, b, c = fourier_setup(n_x=4096, n_max=100)
    s = fourier_solve.all_partial_sums(b, c)
    p = fourier_solve.decay_rate(fourier_solve.errors(f, s, w))
    assert torch.allclose(p, torch.tensor([1.0, 1.0, 3.0], dtype=torch.float64), atol=0.1)


# --- numerical integration -----------------------------------------------------------------------------------------


def test_gauss_legendre_is_exact_for_degree_2n_minus_1_and_weights_sum_to_2():
    for n in (2, 3, 5, 8, 12):
        x, w = quad.gauss_legendre(quad_tensors.jacobi_matrix(n, CPU))
        assert abs(w.sum().item() - 2) < 1e-13
        degrees = torch.arange(2 * n, dtype=torch.float64)  # 0 … 2N − 1
        monomials = x[None, :] ** degrees[:, None]  # (2N, N)
        exact = torch.where(degrees % 2 == 0, 2 / (degrees + 1), torch.zeros_like(degrees))
        assert torch.allclose(quad.integral(w, monomials), exact, atol=1e-13)
        # degree 2N is no longer exact
        assert abs(quad.integral(w, x[None] ** (2 * n)).item() - 2 / (2 * n + 1)) > 1e-10


def test_gauss_legendre_matches_numpy_leggauss():
    import numpy as np

    x, w = quad.gauss_legendre(quad_tensors.jacobi_matrix(10, CPU))
    xr, wr = np.polynomial.legendre.leggauss(10)
    assert np.allclose(x.numpy(), xr, atol=1e-13) and np.allclose(w.numpy(), wr, atol=1e-13)


def test_measured_convergence_orders_are_2_and_4_and_drop_for_kink_and_singularity():
    ns = quad_tensors.n_values(256)
    err = quad_solve.errors(quad_solve.all_rules(ns, CPU), kink=0.577, alpha=0.5)
    p = quad_solve.orders(ns, err)  # (method, integrand)
    assert abs(p[0, 0] - 2) < 0.1  # trapezoid, smooth
    assert abs(p[1, 0] - 4) < 0.15  # Simpson, smooth
    assert err[2, :, 0].min() < 1e-14  # Gauss reaches round-off on the smooth integrand
    assert abs(p[1, 1] - 2) < 0.3 and abs(p[2, 1] - 2) < 0.3  # a kink drops Simpson and Gauss to order 2
    assert abs(p[0, 2] - 1.5) < 0.1 and abs(p[1, 2] - 1.5) < 0.1  # (1+x)^½: order 1 + α
    assert abs(p[2, 2] - 3) < 0.15  # Gauss: order 2 + 2α


def test_trapezoid_and_simpson_weights_sum_to_interval_length():
    for rule in (quad.trapezoid, quad.simpson):
        x, w = rule(16, CPU)
        assert abs(w.sum().item() - 2) < 1e-14
        assert abs(quad.integral(w, x[None] ** 3).item()) < 1e-14  # odd functions integrate to 0


# --- linear maps ---------------------------------------------------------------------------------------------------


def test_det_of_product_is_product_of_dets():
    eps = maps_tensors.levi_civita(CPU)
    a = torch.randn(50, 2, 2, dtype=torch.float64, generator=GEN)
    b = torch.randn(50, 2, 2, dtype=torch.float64, generator=GEN)
    assert torch.allclose(maps.det(a @ b, eps), maps.det(a, eps) * maps.det(b, eps), atol=1e-12)
    assert torch.allclose(maps.det(a, eps), torch.linalg.det(a), atol=1e-12)


def test_abs_det_is_area_of_image_of_unit_square():
    eps = maps_tensors.levi_civita(CPU)
    a = torch.randn(20, 2, 2, dtype=torch.float64, generator=GEN)
    corners = maps.apply(a, maps_tensors.unit_square(CPU))[:, :4]  # (20, 4, 2)
    x, y = corners[..., 0], corners[..., 1]
    shoelace = 0.5 * (x * y.roll(-1, dims=1) - x.roll(-1, dims=1) * y).sum(dim=1)  # signed area
    assert torch.allclose(shoelace, maps.det(a, eps), atol=1e-12)


def test_eigenvectors_stay_on_their_line_along_the_whole_path():
    a = torch.tensor([[2.0, 1.0], [0.5, 1.5]], dtype=torch.float64)
    lam, vec = maps_solve.real_eigenvectors(a)
    assert torch.allclose(lam.sort().values, torch.tensor([1.0, 2.5], dtype=torch.float64))
    m = maps.interpolate(a, torch.linspace(0, 1, 11, dtype=torch.float64))  # (11, 2, 2)
    moved = maps.apply(m, vec.T)  # (11, 2 eigenvectors, 2)
    cross = moved[..., 0] * vec.T[:, 1] - moved[..., 1] * vec.T[:, 0]  # zero ⇔ parallel to the original
    assert cross.abs().max() < 1e-12
    # a rotation has no real eigenvectors
    lam_r, _ = maps_solve.real_eigenvectors(torch.tensor(maps_tensors.MATRICES["rotation"][0], dtype=torch.float64))
    assert lam_r.numel() == 0


def test_image_of_unit_circle_has_semi_axes_equal_to_singular_values():
    a = torch.randn(30, 2, 2, dtype=torch.float64, generator=GEN)
    ellipse = maps.apply(a, maps_tensors.unit_circle(20001, CPU))  # (30, K, 2)
    _, sigma = maps_solve.singular(a)
    assert torch.allclose(maps_solve.circle_axes(ellipse), sigma, rtol=1e-5, atol=1e-6)


def test_negative_det_flips_orientation_and_path_crosses_zero():
    eps = maps_tensors.levi_civita(CPU)
    a, _ = maps_tensors.matrices("reflection", [0, 0, 0, 0], CPU)
    m, _, _ = maps_solve.path(a, frames=41, hold=0)
    d = maps.det(m, eps)
    assert d[0] == 1 and abs(d[-1] + 1) < 1e-12 and d.min() < 0 < d.max()
