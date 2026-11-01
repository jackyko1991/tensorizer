"""Animate the wave field behind the slits next to the intensity pattern building up on the screen."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .tensors import BARRIER_X, NX, NY, SPONGE, Setup


def animate(
    times: torch.Tensor, frames: torch.Tensor, intensity: torch.Tensor, s: Setup, fringes: list[float], path: Path
) -> Path:
    u = frames[:, SPONGE:-SPONGE, SPONGE:-SPONGE].cpu().numpy()  # (S, ny, nx)
    screen = intensity[:, SPONGE:-SPONGE].cpu().numpy()  # (S, ny)
    wall = s.wall[SPONGE:-SPONGE, SPONGE:-SPONGE].cpu().numpy()
    t = times.cpu().numpy()
    # coordinates relative to the centre of the slits
    x0, x1 = SPONGE - BARRIER_X, NX - SPONGE - BARRIER_X
    y0, y1 = SPONGE - NY / 2, NY - SPONGE - NY / 2
    y = np.arange(u.shape[1]) + y0
    behind = u[len(u) // 2 :, :, BARRIER_X - SPONGE + 5 :]  # transmitted waves, second half of the run
    vmax = np.abs(behind).max() * 0.7 + 1e-12  # the strong incident wave simply saturates the colour map

    fig, (ax, ax_i) = plt.subplots(1, 2, figsize=(7, 4), sharey=True, gridspec_kw={"width_ratios": [3, 1]})
    field = ax.imshow(
        u[0],
        origin="lower",
        extent=(x0, x1, y0, y1),
        cmap=plt.get_cmap("RdBu_r", 11),
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.imshow(np.ma.masked_where(~wall, wall), origin="lower", extent=(x0, x1, y0, y1), cmap="binary", vmin=0, vmax=1)
    screen_x = s.screen_x - BARRIER_X
    ax.axvline(screen_x, ls="--", color="#111827", lw=1)
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.grid(False)
    title = ax.set_title("")

    for yf in [f for f in fringes if y0 < f < y1]:  # where the textbook formula puts the bright fringes
        ax_i.axhline(yf, color="#9ca3af", lw=0.8, ls=":")
    (curve,) = ax_i.plot(screen[0], y, color="#dc2626")
    ax_i.set_xlim(0, screen.max() * 1.05 + 1e-12)
    ax_i.set_xticks([])
    ax_i.set_ylim(y0, y1)
    ax_i.set_xlabel("intensity ⟨u²⟩")
    ax_i.set_title("screen")

    def update(k: int):
        field.set_data(u[k])
        curve.set_xdata(screen[k])
        title.set_text(f"wave field u(x, y), t = {t[k]:.0f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 72), path, fps=12)
