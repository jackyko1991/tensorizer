"""Checks for the differentiable-rendering problems in problems/graphics: Gaussian splatting, NeRF volume rendering
and bundle adjustment."""

import math

import torch

from problems.graphics.bundle_adjustment import equation as ba_eq
from problems.graphics.bundle_adjustment import solve as ba_solve
from problems.graphics.bundle_adjustment import tensors as ba_tz
from problems.graphics.gaussian_splatting import equation as gs_eq
from problems.graphics.gaussian_splatting import solve as gs_solve
from problems.graphics.gaussian_splatting import tensors as gs_tz
from problems.graphics.nerf_volume_rendering import equation as nf_eq
from problems.graphics.nerf_volume_rendering import tensors as nf_tz

CPU = torch.device("cpu")

# --- Gaussian splatting ----------------------------------------------------------------------------------


def test_covariance_from_quaternion_and_scale_is_spd():
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(50, 4, generator=gen, dtype=torch.float64)
    log_s = torch.randn(50, 3, generator=gen, dtype=torch.float64)
    R = gs_eq.quat_to_rotmat(q)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(50, 3, 3), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(50, dtype=torch.float64), atol=1e-12)
    cov = gs_eq.covariance_3d(q, log_s)
    assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-12)
    ev = torch.linalg.eigvalsh(cov)
    assert (ev > 0).all()
    assert torch.allclose(ev, torch.sort(torch.exp(2 * log_s), dim=1).values, rtol=1e-9)  # eigenvalues = s²


def test_projected_covariance_matches_finite_difference_jacobian():
    cams = gs_tz.training_cameras(3, 32, CPU)
    W, t = cams["W"].double(), cams["t"].double()
    gen = torch.Generator().manual_seed(1)
    mean = 0.5 * torch.randn(4, 3, generator=gen, dtype=torch.float64)
    quat = torch.randn(4, 4, generator=gen, dtype=torch.float64)
    cov3 = gs_eq.covariance_3d(quat, torch.log(0.05 + 0.2 * torch.rand(4, 3, generator=gen, dtype=torch.float64)))
    x_cam = gs_eq.to_camera(mean, W, t)
    cov2 = gs_eq.covariance_2d(cov3, W, gs_eq.projection_jacobian(x_cam, cams["f"])) - gs_eq.LOW_PASS * torch.eye(2)

    def pixel(x_world, v):  # world point → pixel in camera v
        return gs_eq.project(W[v] @ x_world + t[v], cams["f"], cams["c"])

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
    alpha = gs_eq.splat_alpha(gs_eq.pixel_features(pix, c), mu2, cov2, opacity, c)
    d = pix[None, None] - mu2[:, :, None]
    q = torch.einsum("vmpa,vmab,vmpb->vmp", d, torch.linalg.inv(cov2), d)
    expected = (opacity[..., None] * torch.exp(-0.5 * q)).clamp(max=gs_eq.ALPHA_MAX)
    assert torch.allclose(alpha, expected, atol=1e-10)


def test_alpha_compositing_weights_and_transmittance():
    gen = torch.Generator().manual_seed(3)
    alpha = gs_eq.ALPHA_MAX * torch.rand(2, 30, 50, generator=gen, dtype=torch.float64)
    colour = torch.rand(2, 30, 3, generator=gen, dtype=torch.float64)
    bg = torch.ones(3, dtype=torch.float64)
    image, w, T = gs_eq.composite(alpha, colour, bg)
    assert (w >= 0).all() and (w.sum(dim=1) <= 1 + 1e-12).all()
    assert (T[:, 1:] <= T[:, :-1] + 1e-15).all() and torch.allclose(T[:, 0], torch.ones_like(T[:, 0]))
    t_final = torch.prod(1 - alpha, dim=1)
    assert torch.allclose(w.sum(dim=1) + t_final, torch.ones_like(t_final))  # light is either absorbed or passes
    # a fully white scene composited on white stays white
    image_white, _, _ = gs_eq.composite(alpha, torch.ones_like(colour), bg)
    assert torch.allclose(image_white, torch.ones_like(image_white))
    assert ((image >= 0) & (image <= 1)).all()


def test_splatting_loss_decreases_on_small_seeded_run():
    torch.manual_seed(0)
    res, n_views = 24, 6
    sc = gs_tz.scene(CPU)
    cams = gs_tz.training_cameras(n_views, res, CPU)
    novel = gs_tz.novel_camera(n_views, res, CPU)
    gt, gt_novel = gs_tz.render_scene(sc, cams), gs_tz.render_scene(sc, novel)
    g = gs_tz.init_gaussians(60, 0, CPU)
    hist = gs_solve.train(g, cams, gt, novel, gt_novel, torch.ones(3), 60, 0.02, "l1", True, 2, 0)
    loss = hist["loss"]
    assert loss[-10:].mean() < 0.6 * loss[:5].mean()
    assert hist["snaps"][-1]["psnr_novel"] > hist["snaps"][0]["psnr_novel"] + 3


# --- NeRF volume rendering -------------------------------------------------------------------------------


def test_volume_rendering_weights_for_uniform_slab():
    sigma0, a, b = 2.0, 1.0, 2.0
    t = torch.arange(0, 400, dtype=torch.float64)[None, :] * 0.01  # samples every 0.01 on [0, 4)
    sigma = torch.where((t >= a - 1e-9) & (t < b - 1e-9), sigma0, 0.0)  # density σ0 on the slab [a, b)
    w, T = nf_eq.render_weights(sigma, t, far=4.0)
    assert math.isclose(w.sum().item(), 1 - math.exp(-sigma0 * (b - a)), rel_tol=1e-9)  # exact for piecewise σ
    expected_T = torch.exp(-sigma0 * (t - a).clamp(0, b - a))
    assert torch.allclose(T, expected_T, atol=1e-9)
    assert (T[:, 1:] <= T[:, :-1]).all()
    # the expected depth inside the slab: E[t] = a + 1/σ0 - (b - a) e^{-σ0(b-a)} / (1 - e^{-σ0(b-a)}), up to O(δ)
    ea = math.exp(-sigma0 * (b - a))
    mean_t = (w * t).sum().item() / w.sum().item()
    assert abs(mean_t - (a + 1 / sigma0 - (b - a) * ea / (1 - ea))) < 0.01


def test_positional_encoding_improves_high_frequency_fit():
    x = torch.linspace(-1, 1, 256)[:, None]
    target = torch.sin(6 * math.pi * x) * torch.cos(2 * math.pi * x)  # (N, 1)
    pts = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)  # along the x axis
    errors = {}
    for L in (0, 6):
        layers = nf_tz.init_mlp(3 + 6 * L, 64, 3, 0, CPU)
        opt = torch.optim.Adam(nf_tz.parameters(layers), lr=3e-3)
        for _ in range(400):
            loss = ((nf_eq.mlp(nf_eq.positional_encoding(pts, L), layers)[:, :1] - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        errors[L] = loss.item()
    assert errors[6] < 0.1 * errors[0]


def test_positional_encoding_shape_and_values():
    x = torch.tensor([[0.25, -0.5, 1.0]])
    g = nf_eq.positional_encoding(x, 3)
    assert g.shape == (1, 3 + 18)
    assert torch.allclose(g[0, :3], x[0])
    assert torch.allclose(g[0, 3:12].reshape(3, 3)[0], torch.sin(math.pi * torch.tensor([1.0, 2.0, 4.0]) * 0.25))
    assert torch.equal(nf_eq.positional_encoding(x, 0), x)


# --- bundle adjustment -----------------------------------------------------------------------------------


def _ba_problem(n_cams=4, n_pts=25, sigma=0.5, seed=0):
    X = ba_tz.true_points(n_pts, seed)
    omega, t = ba_tz.true_cameras(n_cams)
    full_true = ba_tz.pack(omega, t, X)
    clean = ba_eq.project(ba_eq.to_camera(omega, t, X), ba_tz.FOCAL, ba_tz.IMAGE / 2)
    gen = torch.Generator().manual_seed(seed + 5)
    observed = clean + sigma * torch.randn(clean.shape, generator=gen, dtype=clean.dtype)
    free = ba_tz.gauge_mask(n_cams, n_pts)
    start = ba_tz.perturb(full_true, n_cams, 5.0, 0.2, 0.2, seed)
    start[~free] = full_true[~free]
    fn = ba_solve.residual_function(start, free, n_cams, observed, ba_tz.FOCAL, ba_tz.IMAGE / 2)
    return fn, start, free, observed, full_true


def test_rodrigues_gives_rotations_and_inverts_log_map():
    gen = torch.Generator().manual_seed(4)
    omega = 0.8 * torch.randn(10, 3, generator=gen, dtype=torch.float64)
    R = ba_eq.rodrigues(omega)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(10, 3, 3), atol=1e-12)
    assert torch.allclose(ba_tz.rotmat_to_axis_angle(R), omega, atol=1e-9)


def test_autograd_jacobian_matches_finite_differences():
    fn, start, free, _, _ = _ba_problem(n_cams=3, n_pts=8)
    theta = start[free]
    J = ba_solve.jacobian(fn, theta)
    h = 1e-6
    J_fd = torch.stack([(fn(theta + h * e) - fn(theta - h * e)) / (2 * h) for e in torch.eye(len(theta)).double()], 1)
    assert J.shape == (2 * 3 * 8, len(theta))
    assert torch.allclose(J, J_fd, rtol=1e-5, atol=1e-5)


def test_gauge_fixing_removes_the_null_space():
    fn, start, free, observed, _ = _ba_problem()
    n_cams = 4
    fn_all = ba_solve.residual_function(start, torch.ones_like(free), n_cams, observed, ba_tz.FOCAL, ba_tz.IMAGE / 2)
    H_all, _ = ba_eq.normal_equations(ba_solve.jacobian(fn_all, start), fn_all(start))
    n = len(start)
    assert torch.linalg.matrix_rank(H_all) == n - 7  # 3 rotations + 3 translations + 1 scale
    H, _ = ba_eq.normal_equations(ba_solve.jacobian(fn, start[free]), fn(start[free]))
    assert H.shape == (n - 7, n - 7)
    assert torch.linalg.matrix_rank(H) == n - 7  # full rank after fixing camera 0 and one translation
    assert torch.linalg.eigvalsh(H).min() > 1e-8 * torch.linalg.eigvalsh(H).max()


def test_schur_complement_step_equals_dense_step():
    fn, start, free, _, _ = _ba_problem()
    theta = start[free]
    H, g = ba_eq.normal_equations(ba_solve.jacobian(fn, theta), fn(theta))
    n_cam = 6 * 4 - 7
    assert torch.allclose(ba_solve.schur_step(H, g, n_cam), ba_solve.dense_step(H, g), atol=1e-9)


def test_bundle_adjustment_reaches_the_noise_floor():
    sigma = 0.5
    fn, start, free, observed, full_true = _ba_problem(sigma=sigma)
    theta0 = start[free]
    theta, hist, _ = ba_solve.levenberg_marquardt(fn, theta0, 6 * 4 - 7, 15, "schur")
    costs = torch.tensor(hist["cost"])
    assert (costs[1:] <= costs[:-1] + 1e-9).all()  # LM never accepts an uphill step
    m, n = observed.numel(), len(theta0)
    floor = sigma * math.sqrt((m - n) / m)
    rms0, rms = ba_eq.rms(fn(theta0)).item(), ba_eq.rms(fn(theta)).item()
    assert rms0 > 20 * floor
    assert abs(rms - floor) < 0.2 * floor
    assert (theta - full_true[free]).abs().max() < 0.1  # close to the truth, in the truth's gauge
