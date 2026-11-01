"""Animate E_z as a heatmap, with the dielectric block and the absorbing border outlined."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from problems._common.plotting import frame_indices, save_gif


def animate(times: torch.Tensor, Ez: torch.Tensor, eps_r: torch.Tensor, border: int, path: Path) -> Path:
    t = times.numpy()  # (F,)
    field = Ez.cpu().numpy()  # (F, Nx, Ny)
    nx, ny = field.shape[1:]
    limit = (
        np.quantile(np.abs(field[-1]), 0.99) + 1e-9
    )  # colour scale from the final wave (16 colours keep the GIF small)

    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(
        field[0].T, origin="lower", cmap=plt.get_cmap("RdBu_r", 16), vmin=-limit, vmax=limit, extent=(0, nx, 0, ny),
        interpolation="nearest",
    )  # fmt: skip
    # the dielectric block: wherever ε_r > 1
    rows, cols = np.nonzero(eps_r.cpu().numpy() > 1)
    if len(rows):
        x0, y0 = rows.min(), cols.min()
        ax.add_patch(Rectangle((x0, y0), rows.max() + 1 - x0, cols.max() + 1 - y0, fill=False, ec="k", lw=1.2))
        ax.text(x0 + 2, cols.max() - 2, f"ε_r = {eps_r.max():g}", va="top", fontsize=9)
    ax.add_patch(
        Rectangle((border, border), nx - 2 * border, ny - 2 * border, fill=False, ec="#9ca3af", ls="--", lw=0.8)
    )
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="E_z", shrink=0.8)
    title = ax.set_title("")

    def update(step: int):
        image.set_data(field[step].T)  # transpose: x runs along the horizontal axis
        title.set_text(f"TM wave E_z  (dashed: absorbing border)   t = {t[step]:.0f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 75), path, fps=15)
