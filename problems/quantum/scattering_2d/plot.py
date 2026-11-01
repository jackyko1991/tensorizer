"""Animate |ψ|² in the plane (left) and the probability collected on the detector screen (right)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(X: torch.Tensor, Y: torch.Tensor, V: torch.Tensor, result: dict, x_screen: float, path: Path) -> Path:
    xs, ys = X[:, 0].cpu().numpy(), Y[0, :].cpu().numpy()
    prob = result["prob"].cpu().numpy()  # (T, Nx, Ny)
    screen = result["screen"].cpu().numpy()  # (T, Ny)
    t = result["times"].cpu().numpy()
    extent = (xs[0], xs[-1], ys[0], ys[-1])

    fig, (ax, ax_s) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [2.2, 1]}, sharey=True)
    # imshow wants rows = vertical axis, so transpose (x, y) → (y, x)
    image = ax.imshow(prob[0].T, origin="lower", extent=extent, cmap="inferno", interpolation="bilinear")
    barrier = np.ma.masked_where(V.cpu().numpy().T == 0, np.ones_like(prob[0].T))
    ax.imshow(barrier, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1.4, alpha=0.9)
    ax.axvline(x_screen, color=ACCENT[5], lw=1, ls="--")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)
    clock = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", color="white", family="monospace")

    (profile,) = ax_s.plot([], [], color=ACCENT[5], lw=1.5)
    fill = [ax_s.fill_betweenx(ys, 0 * ys, color=ACCENT[5], alpha=0.25)]
    ax_s.set_xlim(0, 1.1)
    ax_s.set_xlabel("screen counts (scaled)")
    ax_s.set_title("detector at dashed line", fontsize=9)

    def update(step: int):
        image.set_data(prob[step].T)
        image.set_norm(PowerNorm(0.5, vmin=0, vmax=prob[step].max()))  # √ scale, normalised per frame
        counts = screen[step] / max(screen[-1].max(), 1e-30)  # (Ny,) scaled to the final maximum
        profile.set_data(counts, ys)
        fill[0].remove()
        fill[0] = ax_s.fill_betweenx(ys, counts, color=ACCENT[5], alpha=0.25)
        clock.set_text(f"t = {t[step]:5.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path)
