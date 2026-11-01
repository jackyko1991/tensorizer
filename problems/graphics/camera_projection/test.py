"""Physics checks, written first (TDD): what any correct pinhole camera code must satisfy.

Run with `pytest problems/graphics/camera_projection/test.py`. Each test is a mathematical invariant, not an
implementation detail: a point at depth z projects to f·y/z, the graphics pipeline equals K[R|t], parallel lines meet at
one vanishing point, and orthographic size does not depend on depth.
"""

import math

import torch

from . import equation, tensors

CPU = torch.device("cpu")


def _camera(fov=50.0, w=320, h=240):
    eye, target = torch.tensor([3.0, 2.5, 6.0]), torch.tensor([0.0, 0.9, 0.0])
    view = equation.look_at(eye, target)
    return eye, target, view, equation.intrinsics(fov, w, h, CPU), equation.extrinsics(view)


def test_look_at_point_projects_to_principal_point():
    _, target, view, k, rt = _camera()
    x = torch.cat([target, torch.ones(1)])[None]
    assert torch.allclose(equation.pinhole(k, rt, x)[0], torch.tensor([160.0, 120.0]), atol=1e-3)
    clip = equation.clip_coords(equation.perspective(50, 320 / 240, 0.1, 100, CPU), view, torch.eye(4), x)
    assert torch.allclose(equation.to_pixels(clip, 320, 240)[0], torch.tensor([160.0, 120.0]), atol=1e-3)


def test_graphics_pipeline_equals_k_rt():
    _, _, view, k, rt = _camera()
    x, _ = tensors.house(CPU)
    m = equation.model_matrix(30, 1.8, 0.9, CPU)
    clip = equation.clip_coords(equation.perspective(50, 320 / 240, 0.1, 100, CPU), view, m, x)
    world = torch.einsum("kl,vl->vk", m, x)
    assert torch.allclose(equation.to_pixels(clip, 320, 240), equation.pinhole(k, rt, world), atol=1e-3)


def test_known_point_in_camera_frame():
    # camera at the origin looking down −z: a point at depth z, height y lands f·y/z above the centre
    view = equation.look_at(torch.zeros(3), torch.tensor([0.0, 0.0, -1.0]))
    k = equation.intrinsics(90, 200, 200, CPU)  # f = 100 px
    uv = equation.pinhole(k, equation.extrinsics(view), torch.tensor([[0.5, 1.0, -4.0, 1.0]]))[0]
    assert torch.allclose(uv, torch.tensor([100 + 100 * 0.5 / 4, 100 - 100 * 1.0 / 4]), atol=1e-4)


def test_points_at_infinity_give_vanishing_points():
    _, _, _, k, rt = _camera()
    k, rt = k.double(), rt.double()  # far-away points need double precision
    d = torch.tensor([-0.3, 0.0, -1.0], dtype=torch.float64)
    vp = equation.vanishing_point(k, rt, d)
    for start in ([0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [-1.0, 1.0, 2.0]):  # three parallel lines
        far = torch.tensor(start, dtype=torch.float64) + 1e7 * d
        uv = equation.pinhole(k, rt, torch.cat([far, torch.ones(1, dtype=torch.float64)])[None])[0]
        assert torch.allclose(uv, vp, atol=1e-3)
    # every horizontal direction vanishes on the same image row (the horizon) for a camera without roll
    rows = [
        equation.vanishing_point(k, rt, torch.tensor([math.cos(a), 0.0, math.sin(a)], dtype=torch.float64))[1]
        for a in (2.0, 2.5, 3.0)
    ]
    assert torch.allclose(torch.stack(rows), rows[0].expand(3), atol=1e-2)


def test_orthographic_keeps_size_independent_of_depth():
    view = equation.look_at(torch.zeros(3), torch.tensor([0.0, 0.0, -1.0]))
    p = equation.orthographic(2.0, 1.0, 0.1, 100, CPU)
    x = torch.tensor([[1.0, 0.0, -3.0, 1.0], [1.0, 0.0, -30.0, 1.0]])
    px = equation.to_pixels(equation.clip_coords(p, view, torch.eye(4), x), 100, 100)
    assert torch.allclose(px[0], px[1])  # orthographic: no 1/z foreshortening
