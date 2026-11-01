"""Physics checks, written first (TDD): what any correct Newton fractal code must satisfy.

Run with `pytest problems/maths/newton_fractal/test.py`. Each test is a mathematical invariant, not an implementation
detail: the roots are roots, Newton's method converges quadratically with the analytic constant |p''/2p'|, and the
basins of z³ − 1 share its threefold symmetry.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_roots_are_roots():
    for name in tensors.POLYNOMIALS:
        c = tensors.coefficients(name, CPU, torch.complex128)
        r = tensors.roots(c)
        assert len(r) == len(c) - 1
        assert equation.poly(r, c).abs().max() < 1e-12  # p(r) = 0


def test_derivative_coefficients():
    c = torch.tensor([2.0, -2.0, 0.0, 1.0], dtype=torch.complex128)  # z³ − 2z + 2
    z = torch.tensor([0.3 + 0.7j, -1.2 + 0.1j], dtype=torch.complex128)
    assert torch.allclose(equation.poly(z, equation.derivative(c)), 3 * z**2 - 2)


@pytest.mark.parametrize("name", ["z3_minus_1", "z4_minus_1", "z3_minus_2z_plus_2"])
def test_pixels_near_each_root_converge_to_it(name):
    c = tensors.coefficients(name, CPU)
    roots = tensors.roots(c)
    for j, r in enumerate(roots.tolist()):
        z0 = tensors.grid(complex(r), 0.1, 16, CPU)  # a small patch around root j
        labels, conv = solve.iterate(z0, c, roots, 15, 1e-4)
        assert (labels[-1] == j).all()
        assert conv.max() <= 6  # quadratic convergence: only a few steps from close by


@pytest.mark.parametrize(("name", "expected"), [("z3_minus_1", 1.0), ("z4_minus_1", 1.5)])
def test_error_ratio_tends_to_p2_over_2p1(name, expected):
    """e_{k+1}/e_k² → |p''(r)/(2p'(r))|: for z^n − 1 at r = 1 this is (n − 1)/2."""
    c = tensors.coefficients(name, CPU)
    errors, root = solve.error_trace(1.3 + 0.2j, c)
    assert abs(root.item() - 1) < 1e-12
    limit = solve.newton_constant(root, c)
    assert limit == pytest.approx(expected, rel=1e-10)
    k = int(((errors > 1e-7) & (errors < 1e-3)).nonzero()[-1])  # close to the root but above round-off
    assert (errors[k + 1] / errors[k] ** 2).item() == pytest.approx(limit, rel=0.01)


def test_basins_have_threefold_symmetry_and_wada_boundary():
    c = tensors.coefficients("z3_minus_1", CPU)
    roots = tensors.roots(c)
    labels, _ = solve.iterate(tensors.grid(0j, 1.5, 120, CPU), c, roots, 40, 1e-4)
    share = torch.stack([(labels[-1] == j).float().mean() for j in range(3)])
    assert share.sum() > 0.99 and share.min() > 0.3  # every basin takes about a third of a centred square
    _, wada = solve.boundary_stats(labels[-1].long(), 3)
    assert wada > 0.4  # most boundary pixels touch all three basins
