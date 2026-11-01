"""Physics checks, written first (TDD): what any correct quadrature code must satisfy.

Run with `pytest problems/maths/numerical_integration/test.py`. Each test is a mathematical invariant, not an
implementation detail: Gauss–Legendre is exact up to degree 2N − 1, every rule integrates a constant exactly, and the
measured convergence orders match theory (2, 4, and lower for a kink or a singularity).
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_gauss_legendre_is_exact_for_degree_2n_minus_1_and_weights_sum_to_2():
    for n in (2, 3, 5, 8, 12):
        x, w = equation.gauss_legendre(tensors.jacobi_matrix(n, CPU))
        assert abs(w.sum().item() - 2) < 1e-13
        degrees = torch.arange(2 * n, dtype=torch.float64)  # 0 … 2N − 1
        monomials = x[None, :] ** degrees[:, None]  # (2N, N)
        exact = torch.where(degrees % 2 == 0, 2 / (degrees + 1), torch.zeros_like(degrees))
        assert torch.allclose(equation.integral(w, monomials), exact, atol=1e-13)
        # degree 2N is no longer exact
        assert abs(equation.integral(w, x[None] ** (2 * n)).item() - 2 / (2 * n + 1)) > 1e-10


def test_gauss_legendre_matches_numpy_leggauss():
    import numpy as np

    x, w = equation.gauss_legendre(tensors.jacobi_matrix(10, CPU))
    xr, wr = np.polynomial.legendre.leggauss(10)
    assert np.allclose(x.numpy(), xr, atol=1e-13) and np.allclose(w.numpy(), wr, atol=1e-13)


def test_measured_convergence_orders_are_2_and_4_and_drop_for_kink_and_singularity():
    ns = tensors.n_values(256)
    err = solve.errors(solve.all_rules(ns, CPU), kink=0.577, alpha=0.5)
    p = solve.orders(ns, err)  # (method, integrand)
    assert abs(p[0, 0] - 2) < 0.1  # trapezoid, smooth
    assert abs(p[1, 0] - 4) < 0.15  # Simpson, smooth
    assert err[2, :, 0].min() < 1e-14  # Gauss reaches round-off on the smooth integrand
    assert abs(p[1, 1] - 2) < 0.3 and abs(p[2, 1] - 2) < 0.3  # a kink drops Simpson and Gauss to order 2
    assert abs(p[0, 2] - 1.5) < 0.1 and abs(p[1, 2] - 1.5) < 0.1  # (1+x)^½: order 1 + α
    assert abs(p[2, 2] - 3) < 0.15  # Gauss: order 2 + 2α


def test_trapezoid_and_simpson_weights_sum_to_interval_length():
    for rule in (equation.trapezoid, equation.simpson):
        x, w = rule(16, CPU)
        assert abs(w.sum().item() - 2) < 1e-14
        assert abs(equation.integral(w, x[None] ** 3).item()) < 1e-14  # odd functions integrate to 0
