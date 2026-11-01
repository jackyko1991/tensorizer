"""Physics checks, written first (TDD): what any correct thin-lens code must satisfy.

Run with `pytest problems/optics/thin_lens_image/test.py`. The thin-lens equation 1/s + 1/s' = 1/f with magnification m
= −s'/s, and all principal rays meeting at the image.
"""

import math

import torch

from . import equation, solve, tensors


def test_thin_lens_equation_and_magnification():
    f, h = 10.0, 2.0
    s = tensors.object_distances(30.0, 4.0, f, 27)
    result = solve.solve(s, h, f, tensors.principal_slopes(h, s, f))
    s_img, m = result["s_img"], result["m"]
    assert torch.allclose(1 / s + 1 / s_img, torch.full_like(s, 1 / f), atol=1e-5)
    assert torch.allclose(m, -s_img / s)
    assert ((s_img > 0) == (s > f)).all()  # real image only beyond the focal length
    i = (s - 2 * f).abs().argmin()
    assert math.isclose(s[i].item(), 2 * f) and math.isclose(m[i].item(), -1.0, rel_tol=1e-5)  # 2f → 2f, m = −1


def test_principal_rays_meet_at_the_image():
    f, h = 10.0, 2.0
    s = torch.tensor([25.0, 15.0, 6.0])  # real, real, virtual
    result = solve.solve(s, h, f, tensors.principal_slopes(h, s, f))
    heights = solve.height_after_lens(result["y"], result["slope_out"], result["s_img"])  # (K, 3)
    assert torch.allclose(heights, result["h_img"][:, None].expand_as(heights), atol=1e-4)
    assert torch.allclose(result["slope_out"][:, 2], torch.zeros(3), atol=1e-6)  # focal ray leaves parallel
    assert torch.allclose(equation.lens_kink(torch.tensor([[0.0]]), torch.tensor([[0.3]]), f), torch.tensor([[0.3]]))
