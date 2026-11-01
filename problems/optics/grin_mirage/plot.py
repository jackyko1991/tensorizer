"""Animate rays leaving the eye and bending over the hot road (left); build up what the eye sees (right)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(path: torch.Tensor, angles: torch.Tensor, seen: torch.Tensor, eye: float, h: float, out: Path) -> Path:
    pos = path.numpy()  # (S, R, 2)
    mrad = angles.numpy() * 1e3  # (R,)
    seen = seen.numpy()  # (R,) NaN = road
    length = pos[0, 0, 0]
    y_top = max(3.5, 1.6 * eye)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(y / y_top) if np.isfinite(y) else "#9ca3af" for y in seen]

    fig, (ax, ax_v) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax.set_xlim(-0.04 * length, 1.03 * length)
    ax.set_ylim(-0.15, y_top)
    ax.set_xlabel("distance x (m)")
    ax.set_ylabel("height y (m)  (exaggerated)")
    ax.axhspan(-0.15, 0, color="#4b5563")  # the road
    ax.axhspan(0, 2 * h, color="#fed7aa", alpha=0.6, zorder=0)  # the hot, low-index air layer (≈ 2 scale heights)
    ax.text(0.98 * length, 0.03, "hot air: lower n", ha="right", va="bottom", fontsize=8, color="#9a3412")
    # the object at x = 0, coloured by height with the same colour map as the rays
    ax.imshow(
        np.linspace(0, 1, 50)[:, None],
        origin="lower",
        extent=(-0.035 * length, -0.005 * length, 0, y_top),
        cmap=cmap,
        aspect="auto",
        zorder=3,
    )
    ax.plot([length], [eye], "o", color="#111827", ms=7, zorder=4)
    ax.text(length, eye + 0.12, "eye", ha="right", va="bottom", fontsize=8)
    lines = [ax.plot([], [], lw=1.1, color=c)[0] for c in colors]
    ax.set_title("rays traced back from the eye")

    ax_v.set_xlim(0, y_top)
    ax_v.set_ylim(mrad.min() - 0.5, mrad.max() + 0.5)
    ax_v.set_xlabel("object height seen (m)")
    ax_v.set_ylabel("viewing angle (mrad)")
    ax_v.set_title("what the eye sees")
    dots = ax_v.scatter([], [], s=18)

    def update(step: int):
        for r, line in enumerate(lines):
            line.set_data(pos[: step + 1, r, 0], pos[: step + 1, r, 1])
        arrived = (pos[step, :, 0] <= 0) & np.isfinite(seen)  # rays that have reached the object
        dots.set_offsets(np.column_stack([seen[arrived], mrad[arrived]]) if arrived.any() else np.empty((0, 2)))
        dots.set_color([c for c, a in zip(colors, arrived) if a])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(pos), 90), out, fps=12)
