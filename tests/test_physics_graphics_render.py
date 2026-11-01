"""Checks for the rendering problems: rasterisation, Monte Carlo path tracing and homography estimation."""

import math

import torch

from problems.graphics.homography_stitching import equation as hom
from problems.graphics.homography_stitching import solve as hom_solve
from problems.graphics.homography_stitching import tensors as hom_tensors
from problems.graphics.path_tracing import equation as pt
from problems.graphics.path_tracing import solve as pt_solve
from problems.graphics.path_tracing import tensors as pt_tensors
from problems.graphics.triangle_rasterisation import equation as rast
from problems.graphics.triangle_rasterisation import solve as rast_solve
from problems.graphics.triangle_rasterisation import tensors as rast_tensors

CPU = torch.device("cpu")
# A camera at the origin looking down +z with unit focal length. Its rows are (right, up, forward); looking along
# +z with y up, "right" is −x in a right-handed world.
FLIP_X = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
UNIT_CAM = {"eye": torch.zeros(3), "R": FLIP_X, "f": 1.0, "c": 0.0}


# --- rasterisation ---------------------------------------------------------------------------------------------------


def test_barycentrics_sum_to_one_and_are_positive_exactly_inside():
    gen = torch.Generator().manual_seed(0)
    tri = torch.rand(5, 3, 2, generator=gen) * 10
    weights = torch.rand(200, 5, 3, generator=gen)
    weights = weights / weights.sum(-1, keepdim=True)  # convex combinations → points inside every triangle
    q = torch.einsum("ptk,tki->pti", weights, tri)
    for t in range(5):
        lam = rast.barycentric(q[:, t], tri[t : t + 1])[:, 0]  # (P, 3)
        assert torch.allclose(lam.sum(-1), torch.ones(200), atol=1e-5)
        assert (lam >= -1e-5).all()
        assert torch.allclose(lam, weights[:, t], atol=1e-4)  # and they are the weights that built the point
    far = torch.tensor([[100.0, 100.0], [-50.0, 3.0]])
    lam_out = rast.barycentric(far, tri)
    assert torch.allclose(lam_out.sum(-1), torch.ones(2, 5), atol=1e-4)
    assert (lam_out.min(-1).values < 0).all()  # outside: at least one weight is negative


def test_zbuffer_keeps_the_nearest_triangle():
    big = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    screen = torch.stack([big, big + 0.5])  # two overlapping, counter-clockwise triangles
    q = torch.tensor([[2.0, 2.0], [8.0, 0.2]])  # the first pixel is covered by both, the second only by triangle 0
    for near in (0, 1):
        z = torch.full((2, 3), 5.0)
        z[near] = 2.0
        tri_id, lam, depth = rast_solve.rasterise(screen, z, torch.ones(2, dtype=torch.bool), q)
        assert tri_id[0] == near and abs(depth[0] - 2.0) < 1e-5
        assert tri_id[1] == 0
        assert torch.allclose(lam.sum(-1), torch.ones(2), atol=1e-5)


def test_back_faces_are_culled_on_a_closed_mesh():
    verts, faces, _ = rast_tensors.icosphere(2)
    p = verts[faces]
    volume = torch.einsum("ti,ti->t", p[:, 0], torch.linalg.cross(p[:, 1], p[:, 2])).sum() / 6
    assert abs(volume - 4 / 3 * math.pi) < 0.4  # counter-clockwise from outside ⇒ positive enclosed volume
    cam = {"eye": torch.tensor([0.0, 0.0, -6.0]), "R": FLIP_X, "f": 1.0, "c": 0.0}
    screen, _ = rast.project(verts, cam)
    front = rast.signed_area(screen[faces]) > 0
    assert 0.35 < front.float().mean() < 0.5  # a bit under half: a sphere seen from a finite distance


def test_perspective_correct_interpolation_follows_the_surface_affine_does_not():
    near_z, far_z = 2.0, 6.0
    corners = torch.tensor([[-1.0, -1.0, near_z], [1.0, -1.0, far_z], [0.0, 1.0, near_z]])
    screen, z = rast.project(corners, UNIT_CAM)
    u = torch.tensor([[[0.0], [1.0], [0.0]]])  # texture coordinate: 0 at the near corner, 1 at the far one
    midpoint = (corners[0] + corners[1]) / 2  # true texture coordinate here: 0.5
    q, _ = rast.project(midpoint[None], UNIT_CAM)
    lam = rast.barycentric(q, screen[None])[:, 0]  # (1, 3)
    persp = rast.perspective_interp(lam, z[None], u)
    affine = rast.affine_interp(lam, u)
    assert abs(persp.item() - 0.5) < 1e-5
    # On screen the 3D midpoint sits at s = z_far/(z_near + z_far) of the way along the edge, so affine
    # interpolation over-shoots: the near half of the texture is squeezed and the far half stretched.
    assert abs(affine.item() - far_z / (near_z + far_z)) < 1e-5
    assert affine.item() > persp.item()
    # The interpolated 1/z is exact too.
    assert abs(1 / rast.inverse_depth(lam[:, None], z[None]).item() - midpoint[2].item()) < 1e-4


# --- path tracing ----------------------------------------------------------------------------------------------------


def test_cosine_sampling_pdf_is_normalised_and_matches_samples():
    theta = torch.linspace(0, math.pi / 2, 2001, dtype=torch.float64)
    integrand = pt.cosine_pdf(torch.cos(theta)) * torch.sin(theta) * 2 * math.pi  # ∫ p dω over the hemisphere
    assert abs(torch.trapezoid(integrand, theta).item() - 1) < 1e-6
    gen = torch.Generator().manual_seed(0)
    n = torch.nn.functional.normalize(torch.randn(200_000, 3, generator=gen), dim=1)
    w = pt.cosine_sample(n, gen)
    cos = (w * n).sum(-1)
    assert torch.allclose(w.norm(dim=1), torch.ones(len(w)), atol=1e-5)
    assert (cos >= -1e-6).all()
    assert abs(cos.mean().item() - 2 / 3) < 0.005  # E[cos θ] = ∫ cos² θ / π dω = 2/3
    assert abs((cos**2).mean().item() - 1 / 2) < 0.005


def test_furnace_white_enclosure_reaches_the_analytic_radiance():
    rho, le = 0.5, 1.0
    scene = pt_tensors.furnace(CPU, rho, le)
    cam = pt_tensors.camera(12, eye=(0.1, -0.2, 0.3))
    fixed, _ = pt_solve.render(scene, cam, 2, torch.Generator().manual_seed(0), 6, roulette=False)
    expected = le * sum(rho**k for k in range(7))  # every path picks up exactly ρ per bounce: zero variance
    assert torch.allclose(fixed, torch.full_like(fixed, expected), atol=1e-4)
    rr, _ = pt_solve.render(scene, cam, 64, torch.Generator().manual_seed(1), 2, roulette=True)
    assert abs(rr.mean().item() - le / (1 - rho)) < 0.03  # Russian roulette is unbiased: L_e/(1 − ρ)


def test_path_traced_noise_falls_as_one_over_root_spp():
    scene = pt_tensors.cornell_box(CPU)
    cam = pt_tensors.camera(32)
    ref, _ = pt_solve.render(scene, cam, 1024, torch.Generator().manual_seed(9), 3, True)
    errs = []
    for spp in (4, 16, 64):
        img, _ = pt_solve.render(scene, cam, spp, torch.Generator().manual_seed(spp), 3, True)
        errs.append(pt_solve.rmse(img[None], ref).item())
    slope = pt_solve.noise_slope(torch.tensor([4.0, 16.0, 64.0]), torch.tensor(errs))
    assert -0.7 < slope < -0.38, (errs, slope)  # three noisy points: a loose window around −1/2


def test_direct_light_only_has_no_colour_bleeding():
    scene = pt_tensors.cornell_box(CPU)
    cam = pt_tensors.camera(32)
    gi, direct = pt_solve.render(scene, cam, 64, torch.Generator().manual_seed(3), 3, True)
    r, c = pt_tensors.pixel_of(cam, [-0.8, 0.3, -1.0])  # white back wall next to the red wall
    g, d = gi[r - 1 : r + 2, c - 1 : c + 2].mean((0, 1)), direct[r - 1 : r + 2, c - 1 : c + 2].mean((0, 1))
    assert abs(d[0] / d[1] - 1) < 1e-4  # a white wall lit by a white light stays white
    assert g[0] / g[1] > 1.1  # light bounced off the red wall tints it


# --- homography ------------------------------------------------------------------------------------------------------


def normalised(H: torch.Tensor) -> torch.Tensor:
    H = H / H.norm()
    return H * torch.sign(H.flatten()[H.flatten().abs().argmax()])


def test_dlt_recovers_the_homography_exactly_from_noise_free_points():
    gen = torch.Generator().manual_seed(0)
    H_true = torch.eye(3, dtype=torch.float64) + 0.2 * torch.randn(3, 3, generator=gen, dtype=torch.float64)
    H_true[2, :2] *= 0.01  # a mild perspective part, as for real pixel coordinates
    x1 = torch.rand(30, 2, generator=gen, dtype=torch.float64) * 200
    x2 = hom.apply_homography(H_true[None], x1)[0]
    for n in (4, 30):  # the minimal case and the over-determined one
        H = hom.dlt(x1[None, :n], x2[None, :n])[0]
        assert torch.allclose(normalised(H), normalised(H_true), atol=1e-9)
        assert hom.transfer_error(H[None], x1, x2).max() < 1e-7


def test_ransac_rejects_outliers_and_recovers_the_homography():
    size = (192, 144)
    Hs, _, H_true = hom_tensors.views(size, 170.0, CPU)
    x1, x2, is_out = hom_tensors.matches(Hs, size, 120, 0.5, 0.4, seed=3)
    H_naive = hom.dlt(x1[None], x2[None])[0]
    H, inliers, counts = hom_solve.ransac(x1, x2, 2000, 3.0, torch.Generator().manual_seed(0))
    assert counts.shape == (2000,)
    assert not (inliers & is_out).any()  # no wrong match survives
    assert (inliers & ~is_out).sum() >= 0.9 * (~is_out).sum()  # and almost all good ones are kept
    assert hom_solve.corner_error(H, H_true, size) < 1.5
    assert hom_solve.corner_error(H_naive, H_true, size) > 10  # least squares alone is ruined by the outliers
