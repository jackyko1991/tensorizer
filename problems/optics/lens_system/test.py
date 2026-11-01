"""Physics checks, written first (TDD): what any correct lens-system code must satisfy.

Run with `pytest problems/optics/lens_system/test.py`. Ray-transfer matrices give the thin-lens equation 1/s + 1/s' =
1/f, have determinant 1 (étendue is conserved) and bring every ray from an object point to the same image point.
"""

import math

import torch

from . import equation, solve, tensors


def test_thin_lens_equation_gives_sharp_image():
    s, f = 30.0, 10.0
    s_img = 1 / (1 / f - 1 / s)  # 1/s + 1/s' = 1/f
    m = equation.chain(equation.free_space(s), equation.thin_lens(f), equation.free_space(s_img))
    assert abs(m[0, 1]) < 1e-4  # B = 0: height independent of launch angle
    assert math.isclose(m[0, 0].item(), -s_img / s, rel_tol=1e-5)  # magnification
    single = equation.chain(equation.free_space(s), equation.thin_lens(f))
    assert math.isclose(equation.image_distance(single).item(), s_img, rel_tol=1e-5)


def test_chain_matches_multi_dot_and_det_is_one():
    for d in (5.0, 23.0, 61.0):
        m = equation.system_matrix(20.0, 10.0, d, 15.0)
        chained = equation.chain(
            equation.free_space(20.0), equation.thin_lens(10.0), equation.free_space(d), equation.thin_lens(15.0)
        )
        assert torch.allclose(m, chained, atol=1e-5)
        assert math.isclose(torch.linalg.det(m).item(), 1.0, abs_tol=1e-5)


def test_two_lens_image_matches_sequential_thin_lens_formula():
    s, f1, f2, d = 20.0, 10.0, 15.0, 60.0
    s1 = 1 / (1 / f1 - 1 / s)  # image of lens 1
    s2 = 1 / (1 / f2 - 1 / (d - s1))  # becomes the object of lens 2
    rays = tensors.object_rays(1.0, 0.05, 5)
    _, image_z, _ = solve.solve(rays, s, f1, f2, torch.tensor([d]), tensors.axis(150.0))
    assert math.isclose(image_z[0].item(), s + d + s2, rel_tol=1e-4)


def test_traced_rays_meet_at_image():
    s, f1, f2, d = 20.0, 10.0, 15.0, 60.0
    rays = tensors.object_rays(1.0, 0.05, 7)
    _, image_z, image_h = solve.solve(rays, s, f1, f2, torch.tensor([d]), tensors.axis(150.0))
    h = solve.ray_heights(image_z, rays, s, f1, d, f2)  # (1, R)
    assert torch.allclose(h, image_h.expand_as(h), atol=1e-3)
