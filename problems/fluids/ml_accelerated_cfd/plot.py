"""Top: two unseen initial conditions, fine vs plain coarse vs learned coarse.
Bottom: error vs time and the training loss."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from problems._common.plotting import frame_indices, save_gif

FINE, PLAIN, LEARNED = "#111827", "#ea580c", "#2563eb"


def animate(
    x_f: torch.Tensor, x_c: torch.Tensor, times: torch.Tensor, fine: torch.Tensor, plain: torch.Tensor,
    learned: torch.Tensor, err_plain: torch.Tensor, err_learned: torch.Tensor, losses: list, path: Path,
) -> Path:  # fmt: skip
    xf, xc, t = x_f.cpu().numpy(), x_c.cpu().numpy(), times.cpu().numpy()
    uf, up, ul = fine.cpu().numpy(), plain.cpu().numpy(), learned.cpu().numpy()
    ep, el = err_plain.mean(1).cpu().numpy(), err_learned.mean(1).cpu().numpy()
    edges = np.append(xc - 0.5 * (xc[1] - xc[0]), xc[-1] + 0.5 * (xc[1] - xc[0]))  # coarse cell edges

    fig = plt.figure(figsize=(7.4, 4.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    curves, titles = [], []
    for b in range(2):  # two held-out runs
        ax = fig.add_subplot(grid[0, b])
        f = ax.plot(xf, uf[0, b], color=FINE, lw=1.2, label=f"fine ({len(xf)} cells)")[0]
        p = ax.stairs(up[0, b], edges, color=PLAIN, lw=1.5, label=f"plain ({len(xc)} cells)")
        q = ax.stairs(ul[0, b], edges, color=LEARNED, lw=1.5, label=f"learned ({len(xc)} cells)")
        lo, hi = uf[:, b].min(), uf[:, b].max()
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.3 * (hi - lo))
        ax.set_xlim(0, edges[-1])
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        if b == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=1)
        titles.append(ax.set_title(" ", fontsize=9))
        curves.append((f, p, q, b))

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(t, ep, color=PLAIN, lw=1.8, label="plain")
    ax.plot(t, el, color=LEARNED, lw=1.8, label="learned")
    ax.axvspan(2.0, t[-1], color="#f3f4f6", lw=0)
    ax.text(2.05, 0.95 * max(ep.max(), el.max()), "beyond training", fontsize=7, color="#6b7280", va="top")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.1 * max(ep.max(), el.max()))
    ax.set_xlabel("time t")
    ax.set_ylabel("RMS error (unseen runs)")
    ax.legend(fontsize=7, loc="upper left")
    cursor = ax.axvline(0, color="#f59e0b", lw=1.2)

    ax = fig.add_subplot(grid[1, 1])
    ax.semilogy(np.arange(1, len(losses) + 1), losses, color="#374151", lw=1.2)
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))  # plain text: mathtext redraws slowly
    ax.set_xlabel("Adam step")
    ax.set_ylabel("unrolled training loss")

    def update(step: int):
        for f, p, q, b in curves:
            f.set_ydata(uf[step, b])
            p.set_data(up[step, b])
            q.set_data(ul[step, b])
            titles[b].set_text(f"unseen run {b + 1},  t = {t[step]:.2f}")
        cursor.set_xdata([t[step]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 60), path, fps=12)
