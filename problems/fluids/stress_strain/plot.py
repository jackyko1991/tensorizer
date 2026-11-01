"""Three panels: deformed plate (exaggerated), stress ellipse with principal axes, and Mohr's circle."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Ellipse

from problems._common.plotting import frame_indices, save_gif


def animate(x: torch.Tensor, result: dict, path: Path) -> Path:
    mid = x.shape[2] // 2  # middle layer of the plate (z = 0)
    X = x[:, :, mid, :2].cpu().numpy()  # (n, n, 2)
    U = result["u"][:, :, :, mid, :2].cpu().numpy()  # (T, n, n, 2)
    sig = result["sigma"].cpu().numpy() / 1e6  # (T, 3, 3) in MPa
    vals = result["in_plane"].cpu().numpy() / 1e6  # (T, 2) in-plane principal stresses
    dirs = result["in_plane_dirs"].cpu().numpy()  # (T, 2, 2) columns = directions
    scale = 0.12 / max(np.abs(U).max(), 1e-12)  # exaggerate so the largest displacement is 12 % of the plate
    smax = max(np.abs(vals).max(), 1e-9)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3.3))
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax1.set_xlim(-0.75, 0.75)
    ax1.set_ylim(-0.75, 0.75)
    for k in range(X.shape[0]):
        ax1.plot(X[k, :, 0], X[k, :, 1], color="#d1d5db", lw=0.8)
        ax1.plot(X[:, k, 0], X[:, k, 1], color="#d1d5db", lw=0.8)
    rows = [ax1.plot([], [], color="#2563eb", lw=1)[0] for _ in range(X.shape[0])]
    cols = [ax1.plot([], [], color="#2563eb", lw=1)[0] for _ in range(X.shape[0])]
    title1 = ax1.set_title(" ", fontsize=9)

    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_title("stress ellipse, principal axes", fontsize=9)
    ellipse = ax2.add_patch(Ellipse((0, 0), 0, 0, fc="#93c5fd", ec="#1e40af", alpha=0.7))
    arrows = []

    centre, radius = sig[:, 0, 0] / 2 + sig[:, 1, 1] / 2, np.hypot((sig[:, 0, 0] - sig[:, 1, 1]) / 2, sig[:, 0, 1])
    lim = (centre + radius).max() * 1.15 + 1e-9
    ax3.set_xlim(-0.35 * lim, lim)
    ax3.set_ylim(-0.7 * lim, 0.7 * lim)
    ax3.set_aspect("equal")
    ax3.axhline(0, color="#9ca3af", lw=0.8)
    ax3.axvline(0, color="#9ca3af", lw=0.8)
    ax3.set_xlabel("σ (MPa)")
    ax3.set_ylabel("τ (MPa)")
    ax3.set_title("Mohr's circle", fontsize=9)
    circle = ax3.add_patch(Circle((0, 0), 0, fill=False, ec="#2563eb", lw=1.8))
    diameter = ax3.plot([], [], "o-", color="#dc2626", ms=4, lw=1)[0]
    ends = ax3.plot([], [], "s", color="#f59e0b", ms=5)[0]

    def update(step: int):
        D = X + scale * U[step]  # deformed positions (n, n, 2)
        for k in range(X.shape[0]):
            rows[k].set_data(D[k, :, 0], D[k, :, 1])
            cols[k].set_data(D[:, k, 0], D[:, k, 1])
        title1.set_text(f"σ_xx = {sig[step, 0, 0]:.0f}, σ_xy = {sig[step, 0, 1]:.0f} MPa  (×{scale:.0f})")

        v, d = vals[step] / smax, dirs[step]
        angle = np.degrees(np.arctan2(d[1, 1], d[0, 1]))  # direction of the larger principal stress
        ellipse.set_width(2 * max(abs(v[1]), 0.02))
        ellipse.set_height(2 * max(abs(v[0]), 0.02))
        ellipse.set_angle(angle)
        for a in arrows:
            a.remove()
        arrows.clear()
        for value, direction in zip(v, d.T):  # red arrows point out for tension, in for compression
            for sign in (1, -1):
                tip, tail = sign * direction * (abs(value) + 0.2), sign * direction * 0.2
                start, end = (tail, tip) if value > 0 else (tip, tail)
                if abs(value) > 0.02:
                    arrows.append(ax2.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": "#dc2626"}))

        c, r = centre[step], radius[step]
        circle.set_center((c, 0))
        circle.set_radius(r)
        diameter.set_data([sig[step, 0, 0], sig[step, 1, 1]], [sig[step, 0, 1], -sig[step, 0, 1]])
        ends.set_data([c - r, c + r], [0, 0])  # principal stresses σ_2, σ_1

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(sig.shape[0], 60), path, fps=12)
