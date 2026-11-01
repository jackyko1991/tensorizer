"""Animate the relaxation curves with moving TR/TE markers next to the phantom image they produce."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .tensors import TISSUES


def animate(
    t_long: torch.Tensor,
    t_short: torch.Tensor,
    Mz: torch.Tensor,
    Mxy: torch.Tensor,
    TR: torch.Tensor,
    TE: torch.Tensor,
    path: torch.Tensor,
    frames: torch.Tensor,
    out: Path,
) -> Path:
    tl, ts = t_long.cpu().numpy(), t_short.cpu().numpy()
    Mz, Mxy = Mz.cpu().numpy(), Mxy.cpu().numpy()  # (S, T), (S, T')
    tr = TR.cpu().numpy()[path[:, 0].numpy()]  # (F,) TR of each frame
    te = TE.cpu().numpy()[path[:, 1].numpy()]  # (F,) TE of each frame
    images = frames.cpu().numpy()  # (F, N, N)

    fig = plt.figure(figsize=(7.4, 4.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    ax_z = fig.add_subplot(grid[0, 0])
    ax_xy = fig.add_subplot(grid[1, 0])
    ax_im = fig.add_subplot(grid[:, 1])

    for s, name in enumerate(TISSUES):
        ax_z.plot(tl, Mz[s], color=ACCENT[s], lw=1.5, label=name)
        ax_xy.plot(ts, Mxy[s], color=ACCENT[s], lw=1.5)
    ax_z.set_xlim(0, tl[-1])
    ax_z.set_ylim(0, 1.05)
    ax_z.set_ylabel("$M_z$ (recovery)")
    ax_z.set_xlabel("t (ms)", labelpad=0)
    ax_z.legend(fontsize=6.5, loc="lower right", ncol=2)
    ax_xy.set_xlim(0, ts[-1])
    ax_xy.set_ylim(0, 1.05)
    ax_xy.set_ylabel("$M_{xy}$ (decay)")
    ax_xy.set_xlabel("t (ms)", labelpad=0)
    tr_line = ax_z.axvline(tr[0], color="#111827", ls="--", lw=1)
    te_line = ax_xy.axvline(te[0], color="#111827", ls="--", lw=1)
    tr_dots = ax_z.scatter(np.zeros(len(TISSUES)), np.zeros(len(TISSUES)), c=ACCENT[: len(TISSUES)], s=18, zorder=3)
    te_dots = ax_xy.scatter(np.zeros(len(TISSUES)), np.zeros(len(TISSUES)), c=ACCENT[: len(TISSUES)], s=18, zorder=3)

    shown = ax_im.imshow(images[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_im.set_xticks([])
    ax_im.set_yticks([])
    ax_im.grid(False)
    title = ax_im.set_title("")

    def weighting(a: float, b: float) -> str:
        if a < 1500 and b < 40:
            return "T1-weighted"
        if a >= 1500 and b < 40:
            return "PD-weighted"
        if a >= 1500 and b >= 60:
            return "T2-weighted"
        return "mixed"

    def update(step: int):
        a, b = tr[step], te[step]
        tr_line.set_xdata([a, a])
        te_line.set_xdata([b, b])
        tr_dots.set_offsets(np.c_[np.full(len(TISSUES), a), Mz[:, np.abs(tl - a).argmin()]])
        te_dots.set_offsets(np.c_[np.full(len(TISSUES), b), Mxy[:, np.abs(ts - b).argmin()]])
        img = images[step]
        shown.set_data(img / max(img.max(), 1e-9))  # window each image to its brightest pixel
        title.set_text(f"{weighting(a, b)}\nTR = {a:.0f} ms, TE = {b:.0f} ms")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(tr), len(tr)), out, fps=12)
