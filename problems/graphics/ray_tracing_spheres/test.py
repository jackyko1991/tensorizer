"""Physics checks, written first (TDD): what any correct ray tracer code must satisfy.

Run with `pytest problems/graphics/ray_tracing_spheres/test.py`. Each test is a mathematical invariant, not an
implementation detail: ray–sphere distances match the quadratic formula, hit points lie on the sphere, shadowed points
get only ambient light, and reflection keeps angle in = angle out.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_ray_sphere_distances_match_analytic_values():
    # rays from the origin along +z against spheres on the z axis, one offset sideways, one behind, one around us
    o = torch.zeros(3, 3)
    d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.6, 0.8]])
    c = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [2.0, 0.0, 5.0]])
    r = torch.tensor([1.0, 1.0, 2.0, 1.0])
    t = equation.sphere_hits(o, d, c, r)  # (3 rays, 4 spheres)
    assert t.shape == (3, 4)
    assert t[0, 0].item() == pytest.approx(4.0)  # front surface at z = 5 − 1
    assert math.isinf(t[0, 1].item())  # sphere behind the camera
    assert t[0, 2].item() == pytest.approx(2.0)  # starting inside: the far root
    assert math.isinf(t[0, 3].item())  # passes 2 units away from a sphere of radius 1
    assert math.isinf(t[2, 0].item())  # oblique ray: discriminant (d·c)² − (|c|² − r²) = 16 − 24 < 0
    d_hit = equation.normalize(torch.tensor([[0.0, 0.1, 1.0]]))
    dc = (d_hit[0] @ c[0]).item()
    expected = dc - math.sqrt(dc**2 - (25 - 1))
    assert equation.sphere_hits(o[:1], d_hit, c[:1], r[:1])[0, 0].item() == pytest.approx(expected, rel=1e-5)


def test_hit_points_lie_on_the_sphere():
    sc = tensors.scene(CPU)
    o, d = tensors.camera_rays(torch.tensor([0.0, 2.2, 7.5]), torch.tensor([0.0, 0.7, 0.0]), 45, 64, 48)
    t = equation.sphere_hits(o, d, sc["centres"], sc["radii"])
    tmin, s = t.min(dim=1)
    hit = torch.isfinite(tmin)
    assert hit.any()
    p = o[hit] + tmin[hit, None] * d[hit]
    dist = torch.linalg.vector_norm(p - sc["centres"][s[hit]], dim=1)
    assert torch.allclose(dist, sc["radii"][s[hit]], atol=1e-4)  # |p − c| = r


def test_shadowed_point_gets_no_direct_light():
    # a point on the floor directly below a sphere, with the light straight above the sphere
    c, r = torch.tensor([[0.5, 1.0, 0.5]]), torch.tensor([0.5])
    light = torch.tensor([0.5, 5.0, 0.5])
    p = torch.tensor([[0.5, 0.0, 0.5], [3.5, 0.0, 0.5]])  # under the sphere / out in the open
    n = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert equation.lit(p, n, light, c, r).tolist() == [False, True]

    # trace camera rays that land on those two floor points (the first passes beside the sphere)
    sc = {"centres": c, "radii": r, "albedo": torch.tensor([[1.0, 0.0, 0.0]]), "mirror": torch.tensor([0.0])}
    o = torch.tensor([[2.5, 2.0, 0.5], [3.5, 3.0, 0.5]])
    rgb, _ = solve.trace(o, equation.normalize(p - o), sc, light, bounces=0, shininess=50)
    ground = solve.checker(p)
    assert torch.allclose(rgb[0], solve.AMBIENT * ground[0], atol=1e-5)  # only ambient light in the shadow
    assert (rgb[1] > solve.AMBIENT * ground[1] + 0.1).all()  # the open floor is lit


def test_reflection_direction_and_length():
    gen = torch.Generator().manual_seed(0)
    d = equation.normalize(torch.randn(100, 3, generator=gen))
    n = equation.normalize(torch.randn(100, 3, generator=gen))
    r = equation.reflect(d, n)
    assert torch.allclose(r, d - 2 * (d * n).sum(1, keepdim=True) * n)
    assert torch.allclose(torch.linalg.vector_norm(r, dim=1), torch.ones(100), atol=1e-5)
    assert torch.allclose((r * n).sum(1), -(d * n).sum(1), atol=1e-5)  # angle in = angle out
    floor = equation.reflect(torch.tensor([[1.0, -1.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]))
    assert floor.tolist() == [[1.0, 1.0, 0.0]]
