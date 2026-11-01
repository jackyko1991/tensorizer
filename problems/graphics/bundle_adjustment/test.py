"""Physics checks, written first (TDD): what any correct bundle adjustment code must satisfy.

Run with `pytest problems/graphics/bundle_adjustment/test.py`. Each test is a mathematical invariant, not an
implementation detail: Rodrigues gives rotations, the 7-dimensional similarity gauge is the only null space, and the fit
reaches the statistical noise floor σ√((m − n)/m).
"""

import math

import torch

from . import equation, solve, tensors


def _problem(n_cams=4, n_pts=25, sigma=0.5, seed=0):
    X = tensors.true_points(n_pts, seed)
    omega, t = tensors.true_cameras(n_cams)
    full_true = tensors.pack(omega, t, X)
    clean = equation.project(equation.to_camera(omega, t, X), tensors.FOCAL, tensors.IMAGE / 2)
    gen = torch.Generator().manual_seed(seed + 5)
    observed = clean + sigma * torch.randn(clean.shape, generator=gen, dtype=clean.dtype)
    free = tensors.gauge_mask(n_cams, n_pts)
    start = tensors.perturb(full_true, n_cams, 5.0, 0.2, 0.2, seed)
    start[~free] = full_true[~free]
    fn = solve.residual_function(start, free, n_cams, observed, tensors.FOCAL, tensors.IMAGE / 2)
    return fn, start, free, observed, full_true


def test_rodrigues_gives_rotations_and_inverts_log_map():
    gen = torch.Generator().manual_seed(4)
    omega = 0.8 * torch.randn(10, 3, generator=gen, dtype=torch.float64)
    R = equation.rodrigues(omega)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(10, 3, 3), atol=1e-12)
    assert torch.allclose(tensors.rotmat_to_axis_angle(R), omega, atol=1e-9)


def test_autograd_jacobian_matches_finite_differences():
    fn, start, free, _, _ = _problem(n_cams=3, n_pts=8)
    theta = start[free]
    J = solve.jacobian(fn, theta)
    h = 1e-6
    J_fd = torch.stack([(fn(theta + h * e) - fn(theta - h * e)) / (2 * h) for e in torch.eye(len(theta)).double()], 1)
    assert J.shape == (2 * 3 * 8, len(theta))
    assert torch.allclose(J, J_fd, rtol=1e-5, atol=1e-5)


def test_gauge_fixing_removes_the_null_space():
    fn, start, free, observed, _ = _problem()
    n_cams = 4
    fn_all = solve.residual_function(start, torch.ones_like(free), n_cams, observed, tensors.FOCAL, tensors.IMAGE / 2)
    H_all, _ = equation.normal_equations(solve.jacobian(fn_all, start), fn_all(start))
    n = len(start)
    assert torch.linalg.matrix_rank(H_all) == n - 7  # 3 rotations + 3 translations + 1 scale
    H, _ = equation.normal_equations(solve.jacobian(fn, start[free]), fn(start[free]))
    assert H.shape == (n - 7, n - 7)
    assert torch.linalg.matrix_rank(H) == n - 7  # full rank after fixing camera 0 and one translation
    assert torch.linalg.eigvalsh(H).min() > 1e-8 * torch.linalg.eigvalsh(H).max()


def test_schur_complement_step_equals_dense_step():
    fn, start, free, _, _ = _problem()
    theta = start[free]
    H, g = equation.normal_equations(solve.jacobian(fn, theta), fn(theta))
    n_cam = 6 * 4 - 7
    assert torch.allclose(solve.schur_step(H, g, n_cam), solve.dense_step(H, g), atol=1e-9)


def test_reaches_the_noise_floor():
    sigma = 0.5
    fn, start, free, observed, full_true = _problem(sigma=sigma)
    theta0 = start[free]
    theta, hist, _ = solve.levenberg_marquardt(fn, theta0, 6 * 4 - 7, 15, "schur")
    costs = torch.tensor(hist["cost"])
    assert (costs[1:] <= costs[:-1] + 1e-9).all()  # LM never accepts an uphill step
    m, n = observed.numel(), len(theta0)
    floor = sigma * math.sqrt((m - n) / m)
    rms0, rms = equation.rms(fn(theta0)).item(), equation.rms(fn(theta)).item()
    assert rms0 > 20 * floor
    assert abs(rms - floor) < 0.2 * floor
    assert (theta - full_true[free]).abs().max() < 0.1  # close to the truth, in the truth's gauge
