"""Physics checks, written first (TDD): what any correct linear-map code must satisfy.

Run with `pytest problems/maths/linear_maps_2d/test.py`. Each test is a mathematical invariant, not an implementation
detail: det(AB) = det A det B, |det| is the area scale factor, eigenvectors stay on their line, and the unit circle maps
to an ellipse with the singular values as semi-axes.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
GEN = torch.Generator().manual_seed(0)


def test_det_of_product_is_product_of_dets():
    eps = tensors.levi_civita(CPU)
    a = torch.randn(50, 2, 2, dtype=torch.float64, generator=GEN)
    b = torch.randn(50, 2, 2, dtype=torch.float64, generator=GEN)
    ab = equation.det(a @ b, eps)
    assert torch.allclose(ab, equation.det(a, eps) * equation.det(b, eps), atol=1e-12)  # det(AB) = det A · det B
    assert torch.allclose(equation.det(a, eps), torch.linalg.det(a), atol=1e-12)


def test_abs_det_is_area_of_image_of_unit_square():
    eps = tensors.levi_civita(CPU)
    a = torch.randn(20, 2, 2, dtype=torch.float64, generator=GEN)
    corners = equation.apply(a, tensors.unit_square(CPU))[:, :4]  # (20, 4, 2)
    x, y = corners[..., 0], corners[..., 1]
    shoelace = 0.5 * (x * y.roll(-1, dims=1) - x.roll(-1, dims=1) * y).sum(dim=1)  # signed area
    assert torch.allclose(shoelace, equation.det(a, eps), atol=1e-12)  # det = signed area scale factor


def test_eigenvectors_stay_on_their_line_along_the_whole_path():
    a = torch.tensor([[2.0, 1.0], [0.5, 1.5]], dtype=torch.float64)
    lam, vec = solve.real_eigenvectors(a)
    assert torch.allclose(lam.sort().values, torch.tensor([1.0, 2.5], dtype=torch.float64))
    m = equation.interpolate(a, torch.linspace(0, 1, 11, dtype=torch.float64))  # (11, 2, 2)
    moved = equation.apply(m, vec.T)  # (11, 2 eigenvectors, 2)
    cross = moved[..., 0] * vec.T[:, 1] - moved[..., 1] * vec.T[:, 0]  # zero ⇔ parallel to the original
    assert cross.abs().max() < 1e-12
    # a rotation has no real eigenvectors
    lam_r, _ = solve.real_eigenvectors(torch.tensor(tensors.MATRICES["rotation"][0], dtype=torch.float64))
    assert lam_r.numel() == 0


def test_image_of_unit_circle_has_semi_axes_equal_to_singular_values():
    a = torch.randn(30, 2, 2, dtype=torch.float64, generator=GEN)
    ellipse = equation.apply(a, tensors.unit_circle(20001, CPU))  # (30, K, 2)
    _, sigma = solve.singular(a)
    assert torch.allclose(solve.circle_axes(ellipse), sigma, rtol=1e-5, atol=1e-6)


def test_negative_det_flips_orientation_and_path_crosses_zero():
    eps = tensors.levi_civita(CPU)
    a, _ = tensors.matrices("reflection", [0, 0, 0, 0], CPU)
    m, _, _ = solve.path(a, frames=41, hold=0)
    d = equation.det(m, eps)
    assert d[0] == 1 and abs(d[-1] + 1) < 1e-12 and d.min() < 0 < d.max()
