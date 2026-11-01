"""Animate the ray fan while lens 2 slides away from lens 1; mark where the image forms."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(
    z: torch.Tensor,
    heights: torch.Tensor,
    image_z: torch.Tensor,
    image_h: torch.Tensor,
    gaps: torch.Tensor,
    s: float,
    h0: float,
    path: Path,
) -> Path:
    zz = z.numpy()  # (Z,)
    h = heights.numpy()  # (F, Z, R)
    zi, hi, d = image_z.numpy(), image_h.numpy(), gaps.numpy()  # (F,) each
    y_max = 4 * h0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(zz[0], zz[-1])
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("height (cm)")
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.annotate("", (0, h0), (0, 0), arrowprops={"arrowstyle": "-|>", "color": "#111827", "lw": 2})

    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, h.shape[2]))
    rays = [ax.plot([], [], lw=1.0, color=c)[0] for c in colors]
    lens_kw = {"arrowstyle": "<|-|>", "color": "#2563eb", "lw": 2, "mutation_scale": 12}
    ax.annotate("", (s, 0.9 * y_max), (s, -0.9 * y_max), arrowprops=lens_kw)
    lens2 = ax.annotate("", (s, 0.9 * y_max), (s, -0.9 * y_max), arrowprops=lens_kw)
    (image,) = ax.plot([], [], lw=2.5, color="#dc2626")
    (tip,) = ax.plot([], [], "o", color="#dc2626")
    ax.text(s, -0.98 * y_max, "L1", ha="center", va="bottom", color="#2563eb", fontsize=9)
    name2 = ax.text(s, -0.98 * y_max, "L2", ha="center", va="bottom", color="#2563eb", fontsize=9)

    def update(frame: int):
        for r, line in enumerate(rays):
            line.set_data(zz, h[frame, :, r])
        lens2.xy = (s + d[frame], 0.9 * y_max)
        lens2.xyann = (s + d[frame], -0.9 * y_max)
        name2.set_x(s + d[frame])
        real = zi[frame] > s + d[frame]  # image behind lens 2
        image.set_data([zi[frame], zi[frame]], [0, hi[frame]])
        image.set_linestyle("-" if real else ":")
        tip.set_data([zi[frame]], [hi[frame]])
        kind = "real" if real else "virtual"
        ax.set_title(f"lens gap d = {d[frame]:.1f} cm  →  {kind} image at z = {zi[frame]:.1f} cm")

    ax.set_title("lens gap")  # placeholder so tight_layout leaves room for the title
    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(d), 90), path, fps=12)
