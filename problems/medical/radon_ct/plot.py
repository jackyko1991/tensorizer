"""Animate the scan: the sinogram fills angle by angle while plain and filtered back-projections form."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    f: torch.Tensor,  # (N, N) true phantom
    x: torch.Tensor,  # (N,)
    theta: torch.Tensor,  # (A,)
    sino: torch.Tensor,  # (A, S)
    plain: torch.Tensor,  # (A, N, N) smeared, unfiltered projections
    filtered: torch.Tensor,  # (A, N, N) smeared, ramp-filtered projections
    counts: list[int],
    errors: list[float],
    final_error: float,
    path: Path,
) -> Path:
    f, x, theta, sino = f.cpu().numpy(), x.cpu().numpy(), theta.cpu().numpy(), sino.cpu().numpy()
    dtheta = np.pi / len(theta)
    plain = np.cumsum(plain.cpu().numpy(), axis=0) * dtheta  # running sums over the angles seen so far
    filtered = np.cumsum(filtered.cpu().numpy(), axis=0) * dtheta
    row = int(np.argmin(np.abs(x - 0.0)))  # horizontal profile through the ventricles
    gray = plt.get_cmap("gray", 32)
    ext = (-1, 1, -1, 1)

    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.0))
    ax_ph, ax_sino, ax_err = axes[0]
    ax_bp, ax_fbp, ax_prof = axes[1]
    for ax in (ax_ph, ax_bp, ax_fbp):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    ax_ph.imshow(f, cmap=gray, vmin=0, vmax=0.5, origin="lower", extent=ext)
    ax_ph.set_title("phantom $f(x, y)$")
    ax_ph.set_xlim(-1, 1)
    ax_ph.set_ylim(-1, 1)
    offsets = np.linspace(-0.8, 0.8, 7)
    ray_lines = [ax_ph.plot([], [], color=ACCENT[1], lw=0.8, alpha=0.8)[0] for _ in offsets]
    (detector,) = ax_ph.plot([], [], color=ACCENT[0], lw=3)

    shown = np.full_like(sino.T, np.nan)  # (S, A), filled column by column
    sino_img = ax_sino.imshow(
        shown,
        cmap=plt.get_cmap("viridis", 32),
        vmin=0,
        vmax=sino.max(),
        origin="lower",
        aspect="auto",
        extent=(0, 180, -1, 1),
    )
    ax_sino.set_xlabel(r"$\theta$ (deg)")
    ax_sino.set_ylabel("$s$")
    ax_sino.set_title(r"sinogram $p(\theta, s)$")
    ax_sino.grid(False)

    ax_err.loglog(counts, errors, "o-", color=ACCENT[0], ms=4)
    ax_err.plot([len(theta)], [final_error], "*", color="#f59e0b", ms=11, label=f"this scan ({len(theta)})")
    ax_err.legend(fontsize=7)
    ax_err.yaxis.set_minor_formatter(plt.NullFormatter())
    ax_err.set_yticks([0.25, 0.5, 1.0], ["0.25", "0.5", "1"])
    ax_err.set_xlabel("number of angles")
    ax_err.set_ylabel("NRMSE of FBP")
    ax_err.set_title("error vs angles")

    bp_img = ax_bp.imshow(plain[-1], cmap=gray, origin="lower", extent=ext)
    ax_bp.set_title("plain back-projection")
    fbp_img = ax_fbp.imshow(filtered[-1], cmap=gray, vmin=0, vmax=0.5, origin="lower", extent=ext)
    ax_fbp.set_title("filtered back-projection")

    ax_prof.plot(x, f[row], color="#374151", lw=1.2, label="true")
    (fbp_line,) = ax_prof.plot([], [], color=ACCENT[0], lw=1.2, label="FBP")
    (bp_line,) = ax_prof.plot([], [], color=ACCENT[1], lw=1.2, ls="--", label="plain BP (scaled)")
    ax_prof.set_xlim(-1, 1)
    ax_prof.set_ylim(-0.2, 1.2)
    ax_prof.set_xlabel("$x$ (row $y = 0$)")
    ax_prof.legend(fontsize=7, loc="upper right")
    ax_prof.set_title("profile")

    def update(k: int):
        th = theta[k]
        d = np.array([np.cos(th), np.sin(th)])  # detector axis s
        r = np.array([-np.sin(th), np.cos(th)])  # ray direction t
        for line, s in zip(ray_lines, offsets):
            a, b = s * d - 1.3 * r, s * d + 1.3 * r
            line.set_data([a[0], b[0]], [a[1], b[1]])
        end = -1.0 * r
        detector.set_data([end[0] - 0.9 * d[0], end[0] + 0.9 * d[0]], [end[1] - 0.9 * d[1], end[1] + 0.9 * d[1]])
        shown[:, : k + 1] = sino[: k + 1].T
        sino_img.set_data(shown)
        bp = plain[k]
        bp_img.set_data(bp)
        bp_img.set_clim(0, max(bp.max(), 1e-9))
        fbp_img.set_data(filtered[k])
        fbp_line.set_data(x, filtered[k][row])
        bp_line.set_data(x, bp[row] / max(bp[row].max(), 1e-9))
        ax_fbp.set_title(f"filtered back-projection ({k + 1}/{len(theta)})")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(theta), 48), path, fps=8)
