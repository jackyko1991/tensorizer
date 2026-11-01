"""Physics checks, written first (TDD): what any correct Gaussian splatting code must satisfy.

Run with `pytest problems/graphics/gaussian_splatting/test.py`. Each test is a mathematical invariant, not an
implementation detail: covariances are symmetric positive definite, the projected covariance is J W Σ Wᵀ Jᵀ, and alpha
compositing conserves light (absorbed + transmitted = 1).
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_covariance_from_quaternion_and_scale_is_spd():
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(50, 4, generator=gen, dtype=torch.float64)
    log_s = torch.randn(50, 3, generator=gen, dtype=torch.float64)
    R = equation.quat_to_rotmat(q)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(50, 3, 3), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(50, dtype=torch.float64), atol=1e-12)
    cov = equation.covariance_3d(q, log_s)
    assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-12)
    ev = torch.linalg.eigvalsh(cov)
    assert (ev > 0).all()
    assert torch.allclose(ev, torch.sort(torch.exp(2 * log_s), dim=1).values, rtol=1e-9)  # eigenvalues = s²


def test_projected_covariance_matches_finite_difference_jacobian():
    cams = tensors.training_cameras(3, 32, CPU)
    W, t = cams["W"].double(), cams["t"].double()
    gen = torch.Generator().manual_seed(1)
    mean = 0.5 * torch.randn(4, 3, generator=gen, dtype=torch.float64)
    quat = torch.randn(4, 4, generator=gen, dtype=torch.float64)
    cov3 = equation.covariance_3d(quat, torch.log(0.05 + 0.2 * torch.rand(4, 3, generator=gen, dtype=torch.float64)))
    x_cam = equation.to_camera(mean, W, t)
    cov2 = equation.covariance_2d(
        cov3, W, equation.projection_jacobian(x_cam, cams["f"])
    ) - equation.LOW_PASS * torch.eye(2)

    def pixel(x_world, v):  # world point → pixel in camera v
        return equation.project(W[v] @ x_world + t[v], cams["f"], cams["c"])

    h = 1e-6
    for v in range(3):
        for m in range(4):
            cols = [(pixel(mean[m] + h * e, v) - pixel(mean[m] - h * e, v)) / (2 * h) for e in torch.eye(3).double()]
            JW_fd = torch.stack(cols, dim=1)  # (2, 3) = d pixel / d world point
            assert torch.allclose(cov2[v, m], JW_fd @ cov3[m] @ JW_fd.T, rtol=1e-6, atol=1e-6)


def test_polynomial_quadratic_form_equals_direct_evaluation():
    gen = torch.Generator().manual_seed(2)
    res, c = 16, 8.0
    ij = torch.arange(res, dtype=torch.float64) + 0.5
    v, u = torch.meshgrid(ij, ij, indexing="ij")
    pix = torch.stack([u.reshape(-1), v.reshape(-1)], -1)
    mu2 = res * torch.rand(2, 5, 2, generator=gen, dtype=torch.float64)
    A = torch.randn(2, 5, 2, 2, generator=gen, dtype=torch.float64)
    cov2 = A @ A.transpose(-1, -2) + 2 * torch.eye(2, dtype=torch.float64)
    opacity = torch.rand(2, 5, generator=gen, dtype=torch.float64)
    alpha = equation.splat_alpha(equation.pixel_features(pix, c), mu2, cov2, opacity, c)
    d = pix[None, None] - mu2[:, :, None]
    q = torch.einsum("vmpa,vmab,vmpb->vmp", d, torch.linalg.inv(cov2), d)
    expected = (opacity[..., None] * torch.exp(-0.5 * q)).clamp(max=equation.ALPHA_MAX)
    assert torch.allclose(alpha, expected, atol=1e-10)


def test_alpha_compositing_weights_and_transmittance():
    gen = torch.Generator().manual_seed(3)
    alpha = equation.ALPHA_MAX * torch.rand(2, 30, 50, generator=gen, dtype=torch.float64)
    colour = torch.rand(2, 30, 3, generator=gen, dtype=torch.float64)
    bg = torch.ones(3, dtype=torch.float64)
    image, w, T = equation.composite(alpha, colour, bg)
    assert (w >= 0).all() and (w.sum(dim=1) <= 1 + 1e-12).all()
    assert (T[:, 1:] <= T[:, :-1] + 1e-15).all() and torch.allclose(T[:, 0], torch.ones_like(T[:, 0]))
    t_final = torch.prod(1 - alpha, dim=1)
    assert torch.allclose(w.sum(dim=1) + t_final, torch.ones_like(t_final))  # light is either absorbed or passes
    # a fully white scene composited on white stays white
    image_white, _, _ = equation.composite(alpha, torch.ones_like(colour), bg)
    assert torch.allclose(image_white, torch.ones_like(image_white))
    assert ((image >= 0) & (image <= 1)).all()


def test_loss_decreases_on_small_seeded_run():
    torch.manual_seed(0)
    res, n_views = 24, 6
    sc = tensors.scene(CPU)
    cams = tensors.training_cameras(n_views, res, CPU)
    novel = tensors.novel_camera(n_views, res, CPU)
    gt, gt_novel = tensors.render_scene(sc, cams), tensors.render_scene(sc, novel)
    g = tensors.init_gaussians(60, 0, CPU)
    hist = solve.train(g, cams, gt, novel, gt_novel, torch.ones(3), 60, 0.02, "l1", True, 2, 0)
    loss = hist["loss"]
    assert loss[-10:].mean() < 0.6 * loss[:5].mean()
    assert hist["snaps"][-1]["psnr_novel"] > hist["snaps"][0]["psnr_novel"] + 3
