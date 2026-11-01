"""Animate the partial sums of the three waves as N grows, with the Gibbs overshoot and the L² error below."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .tensors import WAVES

COLORS = ["#2563eb", "#dc2626", "#16a34a"]
GIBBS = 0.0895


def animate(
    x: torch.Tensor, f: torch.Tensor, s: torch.Tensor, over: torch.Tensor, err: torch.Tensor, path: Path
) -> Path:
    """x (X,), f (3, X), s (N, 3, X), over (N, 3), err (N, 3)."""
    x, f, s = x[::2].cpu().numpy(), f[:, ::2].cpu().numpy(), s[:, :, ::2].cpu().numpy()  # every 2nd point is enough
    over, err = over.cpu().numpy(), err.cpu().numpy()
    n_max = s.shape[0]
    ns = np.arange(1, n_max + 1)
    odd = ns % 2 == 1  # plot odd N only: the square wave's even harmonics are 0, so its curves would zigzag

    fig = plt.figure(figsize=(9, 5.6))
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.1])
    tops = [fig.add_subplot(grid[0, 2 * k : 2 * k + 2]) for k in range(3)]
    ax_g, ax_e = fig.add_subplot(grid[1, :3]), fig.add_subplot(grid[1, 3:])

    curves = []
    for k, ax in enumerate(tops):
        ax.plot(x, f[k], color="#9ca3af", lw=1.2)
        (line,) = ax.plot(x, s[0, k], color=COLORS[k], lw=1.2)
        curves.append(line)
        if k < 2:  # the top of the overshoot that never goes away: 1 + 0.0895 × jump
            ax.axhline(1 + 2 * GIBBS, color="#f59e0b", lw=1, ls="--")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xticks([-np.pi, 0, np.pi], ["−π", "0", "π"])
        ax.set_title(WAVES[k], fontsize=10)
    title = fig.suptitle("", fontsize=11)

    trails_g, trails_e = [], []
    for k in range(3):
        if k < 2:
            (tg,) = ax_g.plot([], [], color=COLORS[k], lw=1.4, label=WAVES[k])
            trails_g.append(tg)
        (te,) = ax_e.plot([], [], color=COLORS[k], lw=1.4, label=WAVES[k])
        trails_e.append(te)
    ax_g.axhline(GIBBS, color="#f59e0b", lw=1.2, ls="--", label="0.0895 (Gibbs)")
    ax_g.axhline(0, color="#374151", lw=0.8)
    ax_g.set_xlim(1, n_max)
    ax_g.set_ylim(-0.01, 0.16)
    ax_g.set_xlabel("harmonics N")
    ax_g.set_ylabel("overshoot / jump")
    ax_g.set_title("overshoot does not go to 0", fontsize=10)
    ax_g.legend(fontsize=8, loc="upper right")

    ref = ns[odd].astype(float)  # reference trends, drawn a little above the last odd-N errors
    last = ref[-1]
    ax_e.loglog(ref, 2 * err[odd][-1, 0] * last / ref, color="#9ca3af", ls="--", lw=1, label="∝ 1/N")
    ax_e.loglog(ref, 2 * err[odd][-1, 2] * (last / ref) ** 3, color="#9ca3af", ls=":", lw=1.2, label="∝ 1/N³")
    ax_e.set_xlim(1, n_max)
    ax_e.set_ylim(max(err[odd].min() / 3, 1e-14), err.max() * 3)
    ax_e.set_xlabel("harmonics N")
    ax_e.set_ylabel("‖f − S_N‖²")
    ax_e.set_title("squared L² error", fontsize=10)
    ax_e.legend(fontsize=8, loc="lower left", ncol=2)

    def update(i: int):
        n = i + 1
        for k in range(3):
            curves[k].set_ydata(s[i, k])
        shown = odd & (ns <= n)
        for k in range(2):
            trails_g[k].set_data(ns[shown], over[shown, k])
        for k in range(3):
            trails_e[k].set_data(ns[shown], err[shown, k])
        title.set_text(f"Fourier partial sums  $S_N = B_{{[:, :N]}}\\,c_{{[:N]}}$,   N = {n}")

    fig.tight_layout()
    frames = frame_indices(n_max, 60)
    return save_gif(fig, update, frames + frames[-1:] * 6, path, fps=8)
