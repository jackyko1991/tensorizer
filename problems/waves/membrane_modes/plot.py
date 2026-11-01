"""Animate the first six modes side by side, each oscillating at its own frequency."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import displacement


def animate(shapes: torch.Tensor, omega: torch.Tensor, inside: torch.Tensor, periods: float, path: Path) -> Path:
    n_modes, n, _ = shapes.shape
    w = omega[:n_modes]
    T1 = 2 * math.pi / w[0].item()  # fundamental period
    times = np.linspace(0.0, periods * T1, 181)
    outside = ~inside.view(n, n).cpu().numpy()
    shapes = shapes / shapes.flatten(1).abs().max(1).values[:, None, None]  # each mode scaled to peak 1 for display
    limit = 1.0
    cmap = plt.get_cmap("RdBu_r", 16).with_extremes(bad="white")  # white = outside the drum

    fig, axes = plt.subplots(2, 3, figsize=(7, 4.6))
    images = []
    for m, ax in enumerate(axes.flat):
        frame = np.ma.array(shapes[m].cpu().numpy(), mask=outside)
        images.append(ax.imshow(frame.T, origin="lower", cmap=cmap, vmin=-limit, vmax=limit, extent=(0, 1, 0, 1)))
        ax.set_title(f"mode {m + 1}:  ω/ω₁ = {(w[m] / w[0]).item():.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    title = fig.suptitle("membrane modes  φ(x, y) cos(ωt)    t = 0.00 T₁", fontsize=11)

    def update(step: int):
        u = displacement(shapes, w, times[step]).cpu().numpy()  # (M, n, n)
        for m, image in enumerate(images):
            image.set_data(np.ma.array(u[m], mask=outside).T)
        title.set_text(f"membrane modes  φ(x, y) cos(ωt)    t = {times[step] / T1:.2f} T₁")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(times), 90), path, fps=15)
