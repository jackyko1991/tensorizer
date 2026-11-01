"""Animate the rays: tips advance at speed c/n and bend at every interface; a dashed line joins the tips."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .tensors import THICKNESS, WIDTH


def animate(times: torch.Tensor, rays: torch.Tensor, n_layer: torch.Tensor, angles: torch.Tensor, path: Path) -> Path:
    pos = rays.cpu().numpy()  # (T, R, 2)
    t = times.cpu().numpy()
    n = n_layer.cpu().tolist()
    degrees = torch.rad2deg(angles).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(-len(n) * THICKNESS, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    shade = plt.get_cmap("Blues")
    for k, nk in enumerate(n):  # one band per layer, darker = optically denser
        ax.axhspan(-(k + 1) * THICKNESS, -k * THICKNESS, color=shade(0.08 + 0.4 * (nk - 1)), zorder=0)
        ax.text(WIDTH - 0.08, -(k + 0.5) * THICKNESS, f"n = {nk:.2f}", ha="right", va="center", fontsize=9)

    colors = plt.get_cmap("plasma")(np.linspace(0, 0.85, len(degrees)))
    lines = [ax.plot([], [], lw=1.4, color=c)[0] for c in colors]
    dots = [ax.plot([], [], "o", ms=3, color=c)[0] for c in colors]
    (front,) = ax.plot([], [], "--", lw=0.8, color="#374151")
    clock = ax.text(0.02, 0.04, "", transform=ax.transAxes, fontsize=9)
    ax.set_title(f"{len(degrees)} rays, {degrees[0]:.0f}°–{degrees[-1]:.0f}° from the vertical")

    def update(step: int):
        for r, (line, dot) in enumerate(zip(lines, dots)):
            line.set_data(pos[: step + 1, r, 0], pos[: step + 1, r, 1])
            dot.set_data([pos[step, r, 0]], [pos[step, r, 1]])
        moving = (pos[step] != pos[max(step - 1, 0)]).any(-1)  # (R,) rays still inside the box
        tips = np.where(moving[:, None], pos[step], np.nan)  # NaN hides frozen rays from the line
        front.set_data(tips[:, 0], tips[:, 1])  # tips reached at equal time ≈ a wavefront
        clock.set_text(f"t = {t[step]:.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path)
