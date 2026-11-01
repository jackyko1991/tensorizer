"""Main loop: true scene → noisy image points → perturbed start → gauge fixing → Levenberg–Marquardt → compare."""

import math
import time
from pathlib import Path

import torch

from problems._common.runtime import cli, fast_mode

from . import plot, solve, tensors
from .equation import normal_equations, project, rms, rodrigues, to_camera

SEED = 0
SHOW_CAM = 2  # the camera whose image is drawn


def run(params: dict, out_dir: Path) -> Path:
    fast = fast_mode()
    n_cams = params["n_cameras"]
    n_pts = 30 if fast else params["n_points"]
    iterations = 6 if fast else params["iterations"]
    sigma = params["noise"]
    f, c = tensors.FOCAL, tensors.IMAGE / 2
    # float64 on the CPU: the problem is small, and Gauss–Newton needs the precision near the optimum.

    X_true = tensors.true_points(n_pts, SEED)  # (P, 3)
    omega_true, t_true = tensors.true_cameras(n_cams)  # (C, 3), (C, 3)
    full_true = tensors.pack(omega_true, t_true, X_true)
    gen = torch.Generator().manual_seed(SEED + 2)
    clean = project(to_camera(omega_true, t_true, X_true), f, c)  # (C, P, 2)
    observed = clean + sigma * torch.randn(clean.shape, generator=gen, dtype=clean.dtype)

    free = tensors.gauge_mask(n_cams, n_pts)
    n_free, n_cam = int(free.sum()), 6 * n_cams - 7
    m = observed.numel()
    full_start = tensors.perturb(
        full_true, n_cams, params["init_rot"], params["init_noise"], params["init_noise"], SEED
    )
    full_start[~free] = full_true[~free]  # the gauge: camera 0 and camera 1's x-translation at their true values
    fn = solve.residual_function(full_start, free, n_cams, observed, f, c)
    print(f"{n_cams} cameras (f = {f:.0f} px) see {n_pts} points: {m} residuals, {6 * n_cams + 3 * n_pts} "
          f"unknowns − 7 gauge = {n_free} free ({n_cam} camera + {3 * n_pts} point)")  # fmt: skip

    # the gauge freedom: without fixing, JᵀJ has a 7-dimensional null space (rotation, translation, scale)
    fn_all = solve.residual_function(full_start, torch.ones_like(free), n_cams, observed, f, c)
    H_all, _ = normal_equations(solve.jacobian(fn_all, full_start), fn_all(full_start))
    ev = torch.linalg.eigvalsh(H_all)
    print(f"all unknowns free: rank JᵀJ = {int(torch.linalg.matrix_rank(H_all))} of {len(ev)}; smallest 8 "
          f"eigenvalues / largest: " + " ".join(f"{e:.0e}" for e in (ev[:8] / ev[-1]).tolist()))  # fmt: skip

    theta0 = full_start[free].clone()
    t0 = time.perf_counter()
    theta, hist, H = solve.levenberg_marquardt(fn, theta0, n_cam, iterations, params["solver"])
    t_lm = time.perf_counter() - t0
    r0, r1 = fn(theta0), fn(theta)
    floor = sigma * math.sqrt((m - n_free) / m)  # expected RMS at the optimum: noise minus fitted degrees of freedom
    print(f"gauge fixed: rank JᵀJ = {int(torch.linalg.matrix_rank(H))} of {n_free} (full rank)")
    print(f"Levenberg–Marquardt ({params['solver']} solve), {iterations} iterations in {t_lm:.2f} s: RMS "
          f"reprojection error {rms(r0):.2f} → {rms(r1):.3f} px (noise floor σ√(1−n/m) = {floor:.3f} px)")  # fmt: skip

    # the two linear solvers agree
    g = torch.einsum("mn,m->n", solve.jacobian(fn, theta0), fn(theta0))
    H0, _ = normal_equations(solve.jacobian(fn, theta0), fn(theta0))
    d_dense, d_schur = solve.dense_step(H0, g), solve.schur_step(H0, g, n_cam)
    print(f"Schur complement: reduced camera system {n_cam}×{n_cam} instead of {n_free}×{n_free}; "
          f"max |Δ_schur − Δ_dense| = {(d_dense - d_schur).abs().max():.1e}")  # fmt: skip

    errs = [_errors(fn, th, free, full_start, full_true, n_cams) for th in hist["theta"]]
    print(f"point error {errs[0][0]:.3f} → {errs[-1][0]:.4f}   ·   camera-centre error {errs[0][1]:.3f} → "
          f"{errs[-1][1]:.4f}  (scene radius 1, cameras {tensors.RADIUS:.0f} away)")  # fmt: skip

    frames = []
    for th in hist["theta"]:
        full = full_start.clone()
        full[free] = th
        omega, t, X = tensors.unpack(full, n_cams)
        frames.append({"X": X, "centre": _centres(omega, t), "R": rodrigues(omega),
                       "proj": project(to_camera(omega, t, X), f, c)[SHOW_CAM], "r": fn(th)})  # fmt: skip
    truth = {"X": X_true, "centre": _centres(omega_true, t_true), "R": rodrigues(omega_true)}
    return plot.animate(frames, truth, observed[SHOW_CAM], hist["cost"], errs, H, n_cam, sigma, floor,
                        SHOW_CAM, out_dir / "bundle_adjustment.gif")  # fmt: skip


def _centres(omega: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Camera centres in the world: C = -Rᵀ t.   (C, 3)."""
    return -torch.einsum("cji,cj->ci", rodrigues(omega), t)


def _errors(fn, theta, free, full_start, full_true, n_cams) -> tuple[float, float]:
    """RMS distance of the points and of the camera centres from the truth (same gauge, so no alignment needed)."""
    full = full_start.clone()
    full[free] = theta
    omega, t, X = tensors.unpack(full, n_cams)
    o_true, t_true, X_true = tensors.unpack(full_true, n_cams)
    e_pts = (X - X_true).norm(dim=1).pow(2).mean().sqrt().item()
    e_cam = (_centres(omega, t) - _centres(o_true, t_true)).norm(dim=1).pow(2).mean().sqrt().item()
    return e_pts, e_cam


if __name__ == "__main__":
    cli(run, __file__)
