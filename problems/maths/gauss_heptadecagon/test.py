"""Mathematical checks, written first (TDD): what any correct heptadecagon code must satisfy.

Run with `pytest problems/maths/gauss_heptadecagon/test.py`. Each test is a fact, not an implementation detail: Gauss's
nested square roots equal cos(2π/17), the 17th roots of unity sum to zero, every Gauss period solves a quadratic whose
coefficients are periods one level up, Richmond's ruler-and-compass steps land on 17 equally spaced points, and the
Gauss–Wantzel list of constructible polygons agrees with Euler's φ(n) being a power of two.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

F64 = torch.float64
CPU = torch.device("cpu")


def test_gauss_nested_radical_is_cos_2pi_over_17():
    value = equation.gauss_formula(F64)  # 16 cos(2π/17) from square roots only, divided by 16
    assert abs(value - torch.cos(torch.tensor(2 * math.pi / 17, dtype=F64))) < 1e-14


def test_gauss_nested_radical_symbolically():
    sp = pytest.importorskip("sympy")
    s17 = sp.sqrt(17)
    expr = (
        -1 + s17 + sp.sqrt(34 - 2 * s17) + 2 * sp.sqrt(17 + 3 * s17 - sp.sqrt(34 - 2 * s17) - 2 * sp.sqrt(34 + 2 * s17))
    ) / 16
    assert abs(sp.N(expr - sp.cos(2 * sp.pi / 17), 60)) < 1e-50


def test_roots_of_unity_sum_to_zero():
    zeta = tensors.roots_of_unity(17, F64, CPU)  # (17,) complex
    assert zeta.sum().abs() < 1e-14
    assert ((zeta.abs() - 1).abs() < 1e-14).all()
    assert (zeta**17 - 1).abs().max() < 1e-12


def test_cyclic_order_and_generators():
    order = tensors.cyclic_order(3)  # 3^j mod 17
    assert order.tolist() == [1, 3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6]
    assert [g for g in range(2, 17) if tensors.is_primitive_root(g)] == [3, 5, 6, 7, 10, 11, 12, 14]
    with pytest.raises(ValueError):
        tensors.cyclic_order(2)  # 2^8 ≡ 1 mod 17: only half the exponents


def test_eight_term_periods():
    zeta = tensors.roots_of_unity(17, F64, CPU)
    eta = equation.periods(zeta, tensors.cyclic_order(3))  # (5, 16), level m in row m, 2^m periods
    e0, e1 = eta[1, 0], eta[1, 1]
    assert abs(e0 + e1 + 1) < 1e-13 and abs(e0 * e1 + 4) < 1e-13
    assert abs(eta[0, 0] + 1) < 1e-13  # the sum of all 16 non-trivial roots
    assert abs(e0.real - (-1 + math.sqrt(17)) / 2) < 1e-13  # η² + η − 4 = 0
    assert (eta[:4].imag.abs() < 1e-13).all()  # periods of even length are real (3^8 ≡ −1 pairs ζ^k with ζ^−k)


def test_every_period_pair_solves_its_quadratic():
    zeta = tensors.roots_of_unity(17, F64, CPU)
    order = tensors.cyclic_order(3)
    eta = equation.periods(zeta, order)
    for m in range(4):
        n = 2**m
        const, coef = equation.product_in_periods(order, m)  # product = const + coef · (level-m periods), exact ints
        a, b = eta[m + 1, :n], eta[m + 1, n : 2 * n]  # the two halves of every level-m period
        assert (a + b - eta[m, :n]).abs().max() < 1e-12
        assert (a * b - (const + coef.to(zeta.dtype) @ eta[m, :n])).abs().max() < 1e-12
    const, coef = equation.product_in_periods(order, 1)
    assert const.tolist() == [0, 0] and coef.tolist() == [[1, 1], [1, 1]]  # 4-term pairs: product η₀ + η₁ = −1


def test_radical_tower_reproduces_the_periods():
    zeta = tensors.roots_of_unity(17, F64, CPU)
    order = tensors.cyclic_order(3)
    eta = equation.periods(zeta, order)
    tower = solve.periods_by_radicals(eta, order)
    assert (tower - eta).abs().max() < 1e-12
    k = torch.arange(1, 9, dtype=F64)
    two_cos = (2 * torch.cos(2 * math.pi * k / 17)).sort().values
    assert torch.allclose(tower[3, :8].real.sort().values, two_cos, atol=1e-12)
    assert (tower[4] - zeta[order]).abs().max() < 1e-12


@pytest.mark.parametrize("g", [5, 6, 14])
def test_period_tree_does_not_depend_on_the_generator(g):
    zeta = tensors.roots_of_unity(17, F64, CPU)
    ref = equation.periods(zeta, tensors.cyclic_order(3))
    other = equation.periods(zeta, tensors.cyclic_order(g))
    for m in range(4):
        n = 2**m
        assert torch.allclose(ref[m, :n].real.sort().values, other[m, :n].real.sort().values, atol=1e-12)


def test_intersection_primitives():
    gen = torch.Generator().manual_seed(0)
    a, b, c, d = torch.randn(4, 2, generator=gen, dtype=F64)
    p = equation.line_line(a, b, c, d)
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]  # noqa: E731
    assert abs(cross(b - a, p - a)) < 1e-12 and abs(cross(d - c, p - c)) < 1e-12
    centre = torch.tensor([0.1, -0.2], dtype=F64)
    pts = equation.line_circle(a, b, centre, 3.0)  # (2, 2)
    assert ((pts - centre).norm(dim=1) - 3).abs().max() < 1e-12
    assert all(abs(cross(b - a, q - a)) < 1e-12 for q in pts)
    pts = equation.circle_circle(torch.zeros(2, dtype=F64), 1.0, torch.tensor([1.0, 0.0], dtype=F64), 1.0)
    assert torch.allclose(pts[:, 0], torch.full((2,), 0.5, dtype=F64))
    assert torch.allclose(pts[:, 1].abs(), torch.full((2,), math.sqrt(3) / 2, dtype=F64))


def test_richmond_construction_gives_a_regular_17_gon():
    con = solve.richmond(F64, CPU)
    v = con["vertices"]  # (17, 2), vertex k at angle 2πk/17
    assert ((v.norm(dim=1) - 1).abs() < 1e-12).all()  # on the unit circle
    side = (v.roll(-1, 0) - v).norm(dim=1)
    assert (side - 2 * math.sin(math.pi / 17)).abs().max() < 1e-12  # equally spaced
    k = torch.arange(17, dtype=F64)
    exact = torch.stack([torch.cos(2 * math.pi * k / 17), torch.sin(2 * math.pi * k / 17)], 1)
    assert (v - exact).abs().max() < 1e-12
    assert abs(con["points"]["N3"][0] - math.cos(6 * math.pi / 17)) < 1e-14  # Richmond's two feet
    assert abs(con["points"]["N5"][0] - math.cos(10 * math.pi / 17)) < 1e-14
    kinds = {s["kind"] for s in con["steps"]}
    assert kinds == {"circle", "line"} and 20 < len(con["steps"]) < 100


def test_constructible_polygons_up_to_30():
    grid = solve.constructibility(30)
    assert grid["n"][grid["fermat"]].tolist() == [3, 4, 5, 6, 8, 10, 12, 15, 16, 17, 20, 24, 30]


def test_phi_power_of_two_iff_gauss_wantzel():
    grid = solve.constructibility(1000)
    assert (grid["phi_test"] == grid["fermat"]).all()
    phi = equation.euler_phi(200)
    assert phi[1:].tolist() == [sum(math.gcd(k, n) == 1 for k in range(1, n + 1)) for n in range(1, 201)]
    assert grid["fermat"][grid["n"] == 255].item() and not grid["fermat"][grid["n"] == 7].item()
