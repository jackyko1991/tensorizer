"""Animate the registration: two scans start misaligned, the best RANSAC hypothesis snaps them together, ICP slides
them into place (error curves on the right), and a panel shows the normals and the FPFH matches."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from problems._common.plotting import save_gif

from . import equation

BLUE, RED, GREEN, GREY, DARK, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#9ca3af", "#374151", "#f59e0b"
HOLD, SNAP = 6, 10  # frames before RANSAC, frames for the RANSAC snap


def _xyz(p: torch.Tensor):
    """World (x, y up, z) → matplotlib (x, z, y) so the cow stands upright."""
    p = p.numpy()
    return p[:, 0], p[:, 2], p[:, 1]


def _log_rotation(R: torch.Tensor) -> torch.Tensor:
    """Axis-angle ω of a rotation (angle < π): |ω| = angle, direction = axis."""
    angle = torch.acos(((torch.trace(R) - 1) / 2).clamp(-1, 1))
    v = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (angle / (2 * torch.sin(angle).clamp(min=1e-12)))


def animate(out: dict, path: Path) -> Path:
    src, tgt = out["src"], out["tgt"]
    R0, t0, inl = out["ransac"]
    b = out["best"]
    plane = out["icp"]["point_to_plane"]["hist"]
    point = out["icp"]["point_to_point"]["hist"]
    n_icp = len(plane["R"]) - 1

    # the motion applied to the source in every frame: identity → RANSAC (interpolated) → ICP iterations
    omega0 = _log_rotation(R0)
    poses = [
        (torch.eye(3, dtype=R0.dtype), torch.zeros(3, dtype=R0.dtype), "1. two scans, unknown relative pose")
    ] * HOLD
    for s in np.linspace(0, 1, SNAP):
        poses.append((equation.rodrigues(s * omega0), s * t0, "2. RANSAC on FPFH matches: best hypothesis"))
    poses += [(R0, t0, "2. RANSAC on FPFH matches: best hypothesis")] * 3
    icp_iter = [-1] * len(poses)
    for k in range(n_icp + 1):
        reps = 3 if k < 4 else 1
        poses += [(plane["R"][k][b], plane["t"][k][b], f"3. point-to-plane ICP, iteration {k}")] * reps
        icp_iter += [k] * reps
    poses += [poses[-1]] * 4
    icp_iter += [n_icp] * 4

    # rotation errors (deg) of every ICP start against the truth, both methods
    def rot_errs(h):
        return torch.stack([equation.rotation_error_deg(Rk, out["R_true"]) for Rk in h["R"]]).numpy()  # (iters+1, B)

    e_plane, e_point = rot_errs(plane), rot_errs(point)
    its = np.arange(n_icp + 1)

    fig = plt.figure(figsize=(10.0, 5.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.9], width_ratios=[1, 1, 1.05])
    ax3 = fig.add_subplot(gs[:, :2], projection="3d")
    ax_e, ax_r = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    ax_m = fig.add_subplot(gs[2, 2], projection="3d")

    # --- main 3D view ---
    A = tgt["P"]
    ax3.scatter(*_xyz(A), s=2, color=BLUE, alpha=0.6, depthshade=False, label="scan A (target)")
    moved = ax3.scatter(*_xyz(src["P"]), s=2, color=RED, alpha=0.6, depthshade=False, label="scan B (source)")
    c, half = A.mean(0), 0.42 * (A.max(0).values - A.min(0).values).max().item()
    lims = [(float(c[i]) - half, float(c[i]) + half) for i in (0, 2, 1)]
    ax3.set_xlim(*lims[0])
    ax3.set_ylim(*lims[1])
    ax3.set_zlim(*lims[2])
    ax3.set_box_aspect((1, 1, 1), zoom=1.0)
    ax3.set_axis_off()
    ax3.legend(fontsize=7, loc="upper left", markerscale=4)
    title = ax3.set_title("", fontsize=10)

    # --- error curves (all starts in grey, the best start in colour) ---
    for ax, data, ylabel in [(ax_e, (e_point, e_plane), "rotation error (°)"),
                             (ax_r, (point["rmse"].numpy(), plane["rmse"].numpy()), "RMSE")]:  # fmt: skip
        for arr, col, name in [(data[0], ORANGE, "point-to-point"), (data[1], GREEN, "point-to-plane")]:
            ax.semilogy(its, np.maximum(arr, 1e-4), color=col, lw=0.6, alpha=0.35)
            ax.semilogy(its, np.maximum(arr[:, b], 1e-4), color=col, lw=1.5, label=name)
        ax.set_xlim(0, n_icp)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.minorticks_off()
    ax_e.axhline(2.0, color=DARK, ls="--", lw=0.8)
    ax_e.text(0.3, 2.0, "2° test", fontsize=6.5, ha="left", va="bottom", color=DARK)
    ax_e.legend(fontsize=6.5, loc="upper right")
    ax_e.set_title(f"ICP from {e_plane.shape[1]} starts vs ground truth", fontsize=8.5)
    ax_r.set_xlabel("ICP iteration", fontsize=7.5)
    cursors = [ax.axvline(0, color=DARK, lw=0.8, alpha=0.0) for ax in (ax_e, ax_r)]

    # --- normals and FPFH matches: source drawn after RANSAC, shifted sideways so the lines are visible ---
    shift = torch.tensor([0.0, 0.0, 0.0], dtype=A.dtype)
    shift[0] = 1.1 * (A[:, 0].max() - A[:, 0].min()).item()
    S = equation.transform(R0, t0, src["P"]) + shift
    NS = src["N"] @ R0.T
    ax_m.scatter(*_xyz(A), s=0.6, color=BLUE, alpha=0.5, depthshade=False)
    ax_m.scatter(*_xyz(S), s=0.6, color=RED, alpha=0.5, depthshade=False)
    gen = torch.Generator().manual_seed(0)
    for P, N, col in [(A, tgt["N"], BLUE), (S, NS, RED)]:
        sel = torch.randperm(len(P), generator=gen)[:250]
        seg = torch.stack([P[sel], P[sel] + 0.05 * N[sel]], 1)[..., [0, 2, 1]]  # (250, 2, 3) short normal lines
        ax_m.add_collection3d(Line3DCollection(seg.numpy(), colors=col, linewidths=0.4))
    i_s, i_t = out["match"]
    order = torch.randperm(len(i_s), generator=gen)[:60]
    lines = torch.stack([S[i_s[order]], A[i_t[order]]], 1)[..., [0, 2, 1]]  # (60, 2, 3) match lines
    good = inl[order].numpy()
    ax_m.add_collection3d(Line3DCollection(lines.numpy()[~good], colors=GREY, linewidths=0.4))
    ax_m.add_collection3d(Line3DCollection(lines.numpy()[good], colors=GREEN, linewidths=0.8))
    ax_m.set_axis_off()
    ax_m.set_title(f"normals and FPFH matches: {int(inl.sum())} of {len(i_s)} RANSAC inliers (green)", fontsize=7.5)
    allm = torch.cat([A, S])
    cm, hm = allm.mean(0), 0.5 * (allm.max(0).values - allm.min(0).values).max().item()
    ax_m.set_xlim(cm[0] - hm, cm[0] + hm)
    ax_m.set_ylim(cm[2] - hm, cm[2] + hm)
    ax_m.set_zlim(cm[1] - hm, cm[1] + hm)
    ax_m.set_box_aspect((1, 1, 1), zoom=1.8)
    ax_m.view_init(elev=25, azim=-90)

    footer = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=8.5)

    def update(f: int):
        R, t, label = poses[f]
        moved._offsets3d = _xyz(equation.transform(R, t, src["P"]))
        ax3.view_init(elev=15, azim=-70 + 1.2 * f)
        title.set_text(label)
        k = icp_iter[f]
        for cur in cursors:
            cur.set_xdata([max(k, 0)] * 2)
            cur.set_alpha(0.0 if k < 0 else 0.8)
        err = float(equation.rotation_error_deg(R, out["R_true"]))
        terr = float((t - out["t_true"]).norm())
        footer.set_text(f"error vs ground truth: rotation {err:.2f}°, translation {terr:.4f} (object size 1)")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_gif(fig, update, list(range(len(poses))), path, fps=6)
