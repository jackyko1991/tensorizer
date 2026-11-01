"""Physics checks, written first (TDD): what any correct rasteriser code must satisfy.

Run with `pytest problems/graphics/triangle_rasterisation/test.py`. Each test is a mathematical invariant, not an
implementation detail: barycentric weights sum to 1, the z-buffer keeps the nearest surface, a closed mesh encloses a
positive volume, and perspective-correct interpolation is exact in 3D.
"""

import math

import torch

from . import equation, solve, tensors

# A camera at the origin looking down +z with unit focal length. Its rows are (right, up, forward); looking along
# +z with y up, "right" is −x in a right-handed world.
FLIP_X = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
UNIT_CAM = {"eye": torch.zeros(3), "R": FLIP_X, "f": 1.0, "c": 0.0}


def test_barycentrics_sum_to_one_and_are_positive_exactly_inside():
    gen = torch.Generator().manual_seed(0)
    tri = torch.rand(5, 3, 2, generator=gen) * 10
    weights = torch.rand(200, 5, 3, generator=gen)
    weights = weights / weights.sum(-1, keepdim=True)  # convex combinations → points inside every triangle
    q = torch.einsum("ptk,tki->pti", weights, tri)
    for t in range(5):
        lam = equation.barycentric(q[:, t], tri[t : t + 1])[:, 0]  # (P, 3)
        assert torch.allclose(lam.sum(-1), torch.ones(200), atol=1e-5)
        assert (lam >= -1e-5).all()
        assert torch.allclose(lam, weights[:, t], atol=1e-4)  # and they are the weights that built the point
    far = torch.tensor([[100.0, 100.0], [-50.0, 3.0]])
    lam_out = equation.barycentric(far, tri)
    assert torch.allclose(lam_out.sum(-1), torch.ones(2, 5), atol=1e-4)
    assert (lam_out.min(-1).values < 0).all()  # outside: at least one weight is negative


def test_zbuffer_keeps_the_nearest_triangle():
    big = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    screen = torch.stack([big, big + 0.5])  # two overlapping, counter-clockwise triangles
    q = torch.tensor([[2.0, 2.0], [8.0, 0.2]])  # the first pixel is covered by both, the second only by triangle 0
    for near in (0, 1):
        z = torch.full((2, 3), 5.0)
        z[near] = 2.0
        tri_id, lam, depth = solve.rasterise(screen, z, torch.ones(2, dtype=torch.bool), q)
        assert tri_id[0] == near and abs(depth[0] - 2.0) < 1e-5
        assert tri_id[1] == 0
        assert torch.allclose(lam.sum(-1), torch.ones(2), atol=1e-5)


def test_back_faces_are_culled_on_a_closed_mesh():
    verts, faces, _ = tensors.icosphere(2)
    p = verts[faces]
    volume = torch.einsum("ti,ti->t", p[:, 0], torch.linalg.cross(p[:, 1], p[:, 2])).sum() / 6
    assert abs(volume - 4 / 3 * math.pi) < 0.4  # counter-clockwise from outside ⇒ positive enclosed volume
    cam = {"eye": torch.tensor([0.0, 0.0, -6.0]), "R": FLIP_X, "f": 1.0, "c": 0.0}
    screen, _ = equation.project(verts, cam)
    front = equation.signed_area(screen[faces]) > 0
    assert 0.35 < front.float().mean() < 0.5  # a bit under half: a sphere seen from a finite distance


def test_perspective_correct_interpolation_follows_the_surface_affine_does_not():
    near_z, far_z = 2.0, 6.0
    corners = torch.tensor([[-1.0, -1.0, near_z], [1.0, -1.0, far_z], [0.0, 1.0, near_z]])
    screen, z = equation.project(corners, UNIT_CAM)
    u = torch.tensor([[[0.0], [1.0], [0.0]]])  # texture coordinate: 0 at the near corner, 1 at the far one
    midpoint = (corners[0] + corners[1]) / 2  # true texture coordinate here: 0.5
    q, _ = equation.project(midpoint[None], UNIT_CAM)
    lam = equation.barycentric(q, screen[None])[:, 0]  # (1, 3)
    persp = equation.perspective_interp(lam, z[None], u)
    affine = equation.affine_interp(lam, u)
    assert abs(persp.item() - 0.5) < 1e-5
    # On screen the 3D midpoint sits at s = z_far/(z_near + z_far) of the way along the edge, so affine
    # interpolation over-shoots: the near half of the texture is squeezed and the far half stretched.
    assert abs(affine.item() - far_z / (near_z + far_z)) < 1e-5
    assert affine.item() > persp.item()
    # The interpolated 1/z is exact too.
    assert abs(1 / equation.inverse_depth(lam[:, None], z[None]).item() - midpoint[2].item()) < 1e-4
