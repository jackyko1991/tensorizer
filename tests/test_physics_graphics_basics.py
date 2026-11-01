import math

import pytest
import torch

from problems.graphics.camera_projection import equation as cam_eq
from problems.graphics.camera_projection import tensors as cam_t
from problems.graphics.image_filtering import equation as img_eq
from problems.graphics.image_filtering import tensors as img_t
from problems.graphics.ray_tracing_spheres import equation as rt_eq
from problems.graphics.ray_tracing_spheres import solve as rt_solve
from problems.graphics.ray_tracing_spheres import tensors as rt_t

CPU = torch.device("cpu")

# --- ray tracing ---


def test_ray_sphere_distances_match_analytic_values():
    # rays from the origin along +z against spheres on the z axis, one offset sideways, one behind, one around us
    o = torch.zeros(3, 3)
    d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.6, 0.8]])
    c = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [2.0, 0.0, 5.0]])
    r = torch.tensor([1.0, 1.0, 2.0, 1.0])
    t = rt_eq.sphere_hits(o, d, c, r)  # (3 rays, 4 spheres)
    assert t.shape == (3, 4)
    assert t[0, 0].item() == pytest.approx(4.0)  # front surface at z = 5 − 1
    assert math.isinf(t[0, 1].item())  # sphere behind the camera
    assert t[0, 2].item() == pytest.approx(2.0)  # starting inside: the far root
    assert math.isinf(t[0, 3].item())  # passes 2 units away from a sphere of radius 1
    assert math.isinf(t[2, 0].item())  # oblique ray: discriminant (d·c)² − (|c|² − r²) = 16 − 24 < 0
    d_hit = rt_eq.normalize(torch.tensor([[0.0, 0.1, 1.0]]))
    dc = (d_hit[0] @ c[0]).item()
    expected = dc - math.sqrt(dc**2 - (25 - 1))
    assert rt_eq.sphere_hits(o[:1], d_hit, c[:1], r[:1])[0, 0].item() == pytest.approx(expected, rel=1e-5)


def test_hit_points_lie_on_the_sphere():
    sc = rt_t.scene(CPU)
    o, d = rt_t.camera_rays(torch.tensor([0.0, 2.2, 7.5]), torch.tensor([0.0, 0.7, 0.0]), 45, 64, 48)
    t = rt_eq.sphere_hits(o, d, sc["centres"], sc["radii"])
    tmin, s = t.min(dim=1)
    hit = torch.isfinite(tmin)
    assert hit.any()
    p = o[hit] + tmin[hit, None] * d[hit]
    dist = torch.linalg.vector_norm(p - sc["centres"][s[hit]], dim=1)
    assert torch.allclose(dist, sc["radii"][s[hit]], atol=1e-4)


def test_shadowed_point_gets_no_direct_light():
    # a point on the floor directly below a sphere, with the light straight above the sphere
    c, r = torch.tensor([[0.5, 1.0, 0.5]]), torch.tensor([0.5])
    light = torch.tensor([0.5, 5.0, 0.5])
    p = torch.tensor([[0.5, 0.0, 0.5], [3.5, 0.0, 0.5]])  # under the sphere / out in the open
    n = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert rt_eq.lit(p, n, light, c, r).tolist() == [False, True]

    # trace camera rays that land on those two floor points (the first passes beside the sphere)
    sc = {"centres": c, "radii": r, "albedo": torch.tensor([[1.0, 0.0, 0.0]]), "mirror": torch.tensor([0.0])}
    o = torch.tensor([[2.5, 2.0, 0.5], [3.5, 3.0, 0.5]])
    rgb, _ = rt_solve.trace(o, rt_eq.normalize(p - o), sc, light, bounces=0, shininess=50)
    ground = rt_solve.checker(p)
    assert torch.allclose(rgb[0], rt_solve.AMBIENT * ground[0], atol=1e-5)  # only ambient light in the shadow
    assert (rgb[1] > rt_solve.AMBIENT * ground[1] + 0.1).all()  # the open floor is lit


def test_reflection_direction_and_length():
    gen = torch.Generator().manual_seed(0)
    d = rt_eq.normalize(torch.randn(100, 3, generator=gen))
    n = rt_eq.normalize(torch.randn(100, 3, generator=gen))
    r = rt_eq.reflect(d, n)
    assert torch.allclose(r, d - 2 * (d * n).sum(1, keepdim=True) * n)
    assert torch.allclose(torch.linalg.vector_norm(r, dim=1), torch.ones(100), atol=1e-5)
    assert torch.allclose((r * n).sum(1), -(d * n).sum(1), atol=1e-5)  # angle in = angle out
    floor = rt_eq.reflect(torch.tensor([[1.0, -1.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]]))
    assert floor.tolist() == [[1.0, 1.0, 0.0]]


# --- pinhole camera ---


def _camera(fov=50.0, w=320, h=240):
    eye, target = torch.tensor([3.0, 2.5, 6.0]), torch.tensor([0.0, 0.9, 0.0])
    view = cam_eq.look_at(eye, target)
    return eye, target, view, cam_eq.intrinsics(fov, w, h, CPU), cam_eq.extrinsics(view)


def test_look_at_point_projects_to_principal_point():
    _, target, view, k, rt = _camera()
    x = torch.cat([target, torch.ones(1)])[None]
    assert torch.allclose(cam_eq.pinhole(k, rt, x)[0], torch.tensor([160.0, 120.0]), atol=1e-3)
    clip = cam_eq.clip_coords(cam_eq.perspective(50, 320 / 240, 0.1, 100, CPU), view, torch.eye(4), x)
    assert torch.allclose(cam_eq.to_pixels(clip, 320, 240)[0], torch.tensor([160.0, 120.0]), atol=1e-3)


def test_graphics_pipeline_equals_k_rt():
    _, _, view, k, rt = _camera()
    x, _ = cam_t.house(CPU)
    m = cam_eq.model_matrix(30, 1.8, 0.9, CPU)
    clip = cam_eq.clip_coords(cam_eq.perspective(50, 320 / 240, 0.1, 100, CPU), view, m, x)
    world = torch.einsum("kl,vl->vk", m, x)
    assert torch.allclose(cam_eq.to_pixels(clip, 320, 240), cam_eq.pinhole(k, rt, world), atol=1e-3)


def test_known_point_in_camera_frame():
    # camera at the origin looking down −z: a point at depth z, height y lands f·y/z above the centre
    view = cam_eq.look_at(torch.zeros(3), torch.tensor([0.0, 0.0, -1.0]))
    k = cam_eq.intrinsics(90, 200, 200, CPU)  # f = 100 px
    uv = cam_eq.pinhole(k, cam_eq.extrinsics(view), torch.tensor([[0.5, 1.0, -4.0, 1.0]]))[0]
    assert torch.allclose(uv, torch.tensor([100 + 100 * 0.5 / 4, 100 - 100 * 1.0 / 4]), atol=1e-4)


def test_points_at_infinity_give_vanishing_points():
    _, _, _, k, rt = _camera()
    k, rt = k.double(), rt.double()  # far-away points need double precision
    d = torch.tensor([-0.3, 0.0, -1.0], dtype=torch.float64)
    vp = cam_eq.vanishing_point(k, rt, d)
    for start in ([0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [-1.0, 1.0, 2.0]):  # three parallel lines
        far = torch.tensor(start, dtype=torch.float64) + 1e7 * d
        uv = cam_eq.pinhole(k, rt, torch.cat([far, torch.ones(1, dtype=torch.float64)])[None])[0]
        assert torch.allclose(uv, vp, atol=1e-3)
    # every horizontal direction vanishes on the same image row (the horizon) for a camera without roll
    rows = [
        cam_eq.vanishing_point(k, rt, torch.tensor([math.cos(a), 0.0, math.sin(a)], dtype=torch.float64))[1]
        for a in (2.0, 2.5, 3.0)
    ]
    assert torch.allclose(torch.stack(rows), rows[0].expand(3), atol=1e-2)


def test_orthographic_keeps_size_independent_of_depth():
    view = cam_eq.look_at(torch.zeros(3), torch.tensor([0.0, 0.0, -1.0]))
    p = cam_eq.orthographic(2.0, 1.0, 0.1, 100, CPU)
    x = torch.tensor([[1.0, 0.0, -3.0, 1.0], [1.0, 0.0, -30.0, 1.0]])
    px = cam_eq.to_pixels(cam_eq.clip_coords(p, view, torch.eye(4), x), 100, 100)
    assert torch.allclose(px[0], px[1])


# --- image filtering ---


def test_separable_gaussian_equals_2d_conv():
    img = img_t.gaussian_noise(img_t.test_image(64, CPU), 0.05, seed=0)
    for sigma in (0.8, 1.5, 3.0):
        g, g2 = img_t.gaussian_1d(sigma, CPU), img_t.gaussian_2d(sigma, CPU)
        assert g.sum().item() == pytest.approx(1.0)
        assert torch.allclose(img_eq.separable(img, g), img_eq.filter2d(img, g2), atol=1e-5)


def test_unfold_einsum_equals_conv2d():
    img = img_t.test_image(48, CPU)
    for k in (img_t.box(5, CPU), img_t.sobel(CPU)[0], img_t.laplacian_of_gaussian(1.2, CPU)):
        assert torch.allclose(img_eq.filter2d_unfold(img, k), img_eq.filter2d(img, k), atol=1e-5)


def test_sobel_response_to_vertical_edge():
    img = torch.zeros(16, 16)
    img[:, 8:] = 1.0  # dark left half, bright right half
    kx, ky = img_t.sobel(CPU)
    gx, gy, mag, angle = img_eq.gradient(img, kx, ky)
    assert torch.allclose(gy, torch.zeros_like(gy))  # nothing changes vertically
    assert torch.allclose(gx[:, 7], torch.full((16,), 4.0))  # (1 + 2 + 1) × step on both sides of the edge
    assert torch.allclose(gx[:, 8], torch.full((16,), 4.0))
    assert gx[:, :6].abs().max() == 0 and gx[:, 10:].abs().max() == 0  # flat regions give zero
    assert torch.allclose(angle[:, 7], torch.zeros(16))  # the gradient points along +x
    assert torch.equal(mag, gx.abs())


def test_fft_convolution_equals_circular_spatial_convolution():
    img = img_t.gaussian_noise(img_t.test_image(40, CPU), 0.1, seed=3)
    asym = torch.arange(15.0).reshape(3, 5)  # not symmetric: checks the flip and the centring
    for k in (img_t.gaussian_2d(1.5, CPU), img_t.sobel(CPU)[1], asym):
        assert torch.allclose(img_eq.filter_fft(img, k), img_eq.filter2d(img, k, mode="circular"), atol=1e-3)


def test_frequency_responses():
    g2 = img_t.gaussian_2d(1.5, CPU)
    h = img_eq.frequency_response(g2, 64)
    assert h[0].item() == pytest.approx(1.0, abs=1e-5)  # weights sum to 1: flat regions keep their value
    assert h[-1].item() < 0.01  # the finest checkerboard is removed
    assert img_eq.frequency_response(img_t.laplacian_of_gaussian(1.5, CPU), 64)[0].abs().item() < 1e-5


def test_median_removes_isolated_salt_and_pepper():
    clean = torch.full((20, 20), 0.5)
    noisy = clean.clone()
    noisy[5, 5], noisy[12, 7] = 1.0, 0.0
    assert torch.allclose(img_eq.median(noisy, 3), clean)
    blurred = img_eq.filter2d(noisy, img_t.gaussian_2d(1.0, CPU))
    assert (blurred - clean).abs().max() > 0.05  # the Gaussian only spreads the outlier
