"""Light rays bending around the black hole, the lensed sky as a source passes behind it, and its light curve."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import point_magnification


def animate(rays: dict, b: torch.Tensor, alpha: torch.Tensor, M: float, sky: dict, path: Path) -> Path:
    pts = rays["paths"].cpu().numpy()  # (T, B, 2)
    b, alpha = b.cpu().numpy(), alpha.cpu().numpy()
    lensed = sky["lensed"].cpu().numpy()  # (F, n, n)
    centres = sky["centres"].cpu().numpy()  # (F, 2)
    mu = sky["mu"].cpu().numpy()
    half = sky["half_width"]
    F = len(centres)

    fig = plt.figure(figsize=(8, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.2, 1])
    ax_r = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[1, 0])
    ax_i = fig.add_subplot(gs[:, 1])

    # --- null geodesics past the hole ---
    ax_r.set_xlim(-25, 25)
    ax_r.set_ylim(-9, 16)
    ax_r.set_aspect("equal", adjustable="box")
    ax_r.add_patch(plt.Circle((0, 0), 2 * M, color="black", zorder=4))
    ax_r.add_patch(plt.Circle((0, 0), 3 * M, fill=False, ls=":", color="#f59e0b", lw=1.2))
    for k in range(pts.shape[1]):
        captured = not np.isfinite(alpha[k])
        ax_r.plot(pts[:, k, 0], pts[:, k, 1], lw=1.2, color="#6b7280" if captured else ACCENT[k % len(ACCENT)])
    ax_r.set_title(f"null geodesics; b < √27 M = {np.sqrt(27) * M:.2f} M are captured", fontsize=8)
    ax_r.set_xlabel("x (GM/c²)", fontsize=8)
    ax_r.tick_params(labelsize=7)

    # --- light curve ---
    x_src = centres[:, 0]
    u = np.linalg.norm(centres, axis=-1)
    ax_l.semilogy(x_src, point_magnification(torch.tensor(u)).numpy(), color="#9ca3af", lw=1, label="point source")
    curve = ax_l.semilogy([], [], color=ACCENT[0], lw=2, label="this source (pixel sum)")[0]
    ax_l.set_xlim(x_src[0], x_src[-1])
    ax_l.set_ylim(0.9, max(mu.max(), 3) * 1.5)
    ax_l.set_xlabel("source position β_x (θ_E)", fontsize=8)
    ax_l.set_ylabel("magnification μ", fontsize=8)
    ax_l.tick_params(labelsize=7)
    ax_l.legend(fontsize=7, loc="upper left")

    # --- the lensed sky ---
    img = ax_i.imshow(lensed[0], origin="lower", extent=(-half, half, -half, half), cmap="magma", vmin=0, vmax=1)
    ring = np.linspace(0, 2 * np.pi, 200)
    ax_i.plot(np.cos(ring), np.sin(ring), ls="--", lw=0.8, color="#93c5fd")
    src = ax_i.plot([], [], "+", color="#93c5fd", ms=10, mew=1.5)[0]
    ax_i.plot([0], [0], "o", color="white", ms=3)
    ax_i.set_xlabel("θ_x (θ_E)", fontsize=8)
    ax_i.set_ylabel("θ_y (θ_E)", fontsize=8)
    ax_i.tick_params(labelsize=7)
    ax_i.grid(False)
    title = ax_i.set_title("", fontsize=9)

    def update(k: int):
        img.set_data(lensed[k])
        src.set_data([centres[k, 0]], [centres[k, 1]])
        curve.set_data(x_src[: k + 1], mu[: k + 1])
        title.set_text(f"lensed sky   μ = {mu[k]:.2f}   (+ true source position)")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(F, 90), path)
