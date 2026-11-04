"""Physics checks, written first (TDD): what any correct large-extra-dimension calculation must satisfy.

Run with `pytest problems/cosmology/hierarchy_extra_dimensions/test.py`. On R^3 x T^n the Newtonian potential is the
image sum (images of the mass around the compact circles) or, equivalently by Poisson summation, the Kaluza-Klein
sum (massive modes with Yukawa tails). It must fall like 1/r^(1+n) at r << R and like 1/r at r >> R, obey Gauss's
law in 3 + n space dimensions, and look like a Yukawa correction with alpha = 2n, lambda = R just outside the
compact space. ADD with n = 2 and M* = 1 TeV puts R near a millimetre; the Randall-Sundrum warp factor for
k r_c = 11.27 is about 1e-16.
"""

import math

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")
NS = [1, 2, 3, 4, 5, 6]


@pytest.fixture(scope="module")
def shells():
    return {n: tensors.shell_counts(n, 3000, CPU) for n in NS}  # r_n(m): lattice vectors with |k|^2 = m


def test_shell_counts_match_explicit_mode_vectors():
    K = 6
    for n in (1, 2, 3):
        k = tensors.mode_vectors(n, K, CPU)  # (N, n) every mode vector in the cube [-K, K]^n
        m2 = torch.einsum("ki,ki->k", k, k)
        explicit = torch.bincount(m2.long(), minlength=K * K + 1)[: K * K + 1]  # complete shells up to K^2
        assert torch.equal(tensors.shell_counts(n, K * K, CPU).long(), explicit)
    r2 = tensors.shell_counts(2, 30, CPU)
    assert r2[[0, 1, 2, 3, 4, 5, 25]].tolist() == [1, 4, 4, 0, 4, 8, 12]  # sums of two squares


def test_image_sum_equals_kk_sum(shells):
    r = torch.linspace(0.6, 3.0, 13, dtype=torch.float64)
    for n in NS:
        img = equation.potential_images(r, n, shells[n])
        kk = equation.potential_kk(r, n, shells[n])
        assert torch.allclose(img, kk, rtol=2e-3)  # Poisson summation: two ways to write one Green's function


def test_kk_sum_crosses_over_from_1_over_r_to_the_1_plus_n_to_1_over_r(shells):
    for n in NS:
        small = torch.tensor([1e-3, 3e-3], dtype=torch.float64)
        phi_s = equation.potential_images(small, n, shells[n])
        assert torch.allclose(
            -phi_s * small ** (n + 1), torch.ones(2, dtype=torch.float64), rtol=1e-3
        )  # G_D M / r^(1+n)
        large = torch.tensor([20.0, 40.0], dtype=torch.float64)
        phi_l = equation.potential_kk(large, n, shells[n])
        assert torch.allclose(
            -phi_l * large / equation.g4(n), torch.ones(2, dtype=torch.float64), rtol=1e-6
        )  # G_4 M / r
        slope = equation.log_slope(lambda x, n=n: equation.potential_images(x, n, shells[n]), torch.tensor([2e-3]))
        assert slope.item() == pytest.approx(-(n + 1), abs=1e-2)
        slope = equation.log_slope(lambda x, n=n: equation.potential_kk(x, n, shells[n]), torch.tensor([30.0]))
        assert slope.item() == pytest.approx(-1.0, abs=1e-6)


def test_gauss_law_flux_in_4_plus_n_dimensions(shells):
    for n in NS:
        total = (n + 1) * equation.sphere_area(
            n + 2
        )  # (D - 2) * area of S^(D-1), D = 3 + n space dimensions, G_D M = 1
        r = torch.tensor([1e-3], dtype=torch.float64)
        flux_small = (
            equation.sphere_area(n + 2)
            * r ** (n + 2)
            * equation.radial_field(lambda x, n=n: equation.potential_images(x, n, shells[n]), r)
        )
        assert flux_small.item() == pytest.approx(total, rel=1e-3)  # through a tiny S^(2+n) around the mass
        R_big = torch.tensor([25.0], dtype=torch.float64)
        flux_big = (
            4
            * math.pi
            * R_big**2
            * (2 * math.pi) ** n
            * equation.radial_field(lambda x, n=n: equation.potential_kk(x, n, shells[n]), R_big)
        )
        assert flux_big.item() == pytest.approx(total, rel=1e-6)  # through S^2 x T^n far away: the same flux


def test_yukawa_fit_gives_alpha_2n_and_lambda_R(shells):
    r = torch.linspace(16.0, 24.0, 25, dtype=torch.float64)  # beyond the next KK level (mass sqrt(2)/R)
    for n in (1, 2, 3):
        alpha, lam = equation.fit_yukawa(r, equation.kk_deviation(r, n, shells[n]))
        assert alpha == pytest.approx(2 * n, rel=0.03)  # 2n lightest KK modes of mass 1/R on a torus
        assert lam == pytest.approx(1.0, rel=0.01)  # range = R


def test_yukawa_with_alpha_zero_is_newton():
    r = torch.logspace(-2, 2, 9, dtype=torch.float64)
    assert torch.allclose(equation.yukawa(r, 0.0, 0.7, g=2.0), -2.0 / r)
    assert torch.allclose(equation.yukawa(r * 1e4, 3.0, 1e-2, g=1.0), -1.0 / (r * 1e4))  # r >> lambda: Newton again


def test_add_radius_n2_one_tev_is_millimetre_and_n1_is_solar_system():
    n = torch.arange(1, 7, dtype=torch.float64)
    M = torch.tensor([1.0, 10.0], dtype=torch.float64)
    R = equation.add_radius_m(n[:, None], M[None, :])  # (6, 2) metres
    assert 1e-4 < R[1, 0].item() < 1e-2  # n = 2, M* = 1 TeV: about a millimetre (ADD 1998)
    assert R[0, 0].item() > 1e11  # n = 1: larger than the Earth's orbit, long excluded
    assert R[5, 0].item() < 1e-12  # n = 6: below a femtometre
    assert (R[:, 1] < R[:, 0]).all()  # a higher fundamental scale needs smaller dimensions
    # M_Pl^2 = M*^(n+2) R^n holds back
    back = M[None, :] ** (n[:, None] + 2) * (R / tensors.HBARC_M) ** n[:, None] * 1e3 ** (n[:, None] + 2)
    assert torch.allclose(back, torch.full_like(back, tensors.M_PL_GEV**2), rtol=1e-10)


def test_randall_sundrum_warp_factor():
    w = equation.rs_warp(torch.tensor([11.27], dtype=torch.float64))
    assert 1e-16 < w.item() < 1e-15  # e^(-pi k r_c): M_Pl -> TeV
    tev = tensors.M_PL_REDUCED_GEV * w.item() / 1e3
    assert 0.5 < tev < 2.0


def test_hierarchy_ratio():
    ratio = equation.weak_over_gravity()  # G_F / G_N in natural units
    assert 1e32 < ratio < 1e34
