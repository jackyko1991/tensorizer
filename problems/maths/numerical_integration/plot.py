"""Animate N growing: the nodes on each integrand (top) and the error curves on log–log axes (bottom)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from . import tensors
from .solve import FLOOR, METHODS

COLORS = ["#2563eb", "#16a34a", "#dc2626"]  # trapezoid, Simpson, Gauss–Legendre


def animate(ns: list[int], rules: list, err: torch.Tensor, kink: float, alpha: float, path: Path) -> Path:
    """ns (M,), rules[method][i] = (nodes, weights), err (3 methods, M, 3 integrands)."""
    err = err.cpu().numpy()
    n = np.array(ns, dtype=float)
    xs = torch.linspace(-1, 1, 400, dtype=torch.float64)
    fs = tensors.integrands(xs, kink, alpha).numpy()  # (3, 400)
    labels = ["smooth  $1/(1+4x^2)$", f"kink  $|x - {kink:g}|$", f"endpoint  $(1+x)^{{{alpha:g}}}$"]

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6), gridspec_kw={"height_ratios": [1, 1.4]})
    nodes_t, nodes_g, trails = [], [], []
    for f in range(3):
        top, bottom = axes[0, f], axes[1, f]
        top.plot(xs.numpy(), fs[f], color="#374151", lw=1.4)
        top.fill_between(xs.numpy(), 0, fs[f], color="#93c5fd", alpha=0.35)
        (nt,) = top.plot([], [], "|", color=COLORS[0], ms=8, mew=1)
        (ng,) = top.plot([], [], "o", color=COLORS[2], ms=3.5)
        nodes_t.append(nt)
        nodes_g.append(ng)
        top.set_xlim(-1.05, 1.05)
        top.set_ylim(-0.12, fs[f].max() * 1.12)
        top.set_title(labels[f], fontsize=9)

        for m in range(3):
            (tr,) = bottom.loglog([], [], "o-", color=COLORS[m], ms=2.5, lw=1.3, label=METHODS[m])
            trails.append(tr)
        scale = err[0, 2, f]  # reference slopes, anchored at the trapezoid error at the third N
        bottom.loglog(n, scale * (n / n[2]) ** -2, color="#9ca3af", ls="--", lw=1, label="$N^{-2}$")
        bottom.loglog(n, err[1, 2, f] * (n / n[2]) ** -4, color="#9ca3af", ls=":", lw=1.2, label="$N^{-4}$")
        bottom.set_xlim(n[0] * 0.8, n[-1] * 1.25)
        bottom.set_ylim(FLOOR / 3, max(1.0, err[:, :, f].max() * 3))
        bottom.set_xlabel("N (intervals, or Gauss nodes)", fontsize=8)
        if f == 0:
            bottom.set_ylabel("|error|")
            bottom.legend(fontsize=7, loc="lower left")
    title = fig.suptitle("", fontsize=11)

    def update(i: int):
        x_t = rules[0][i][0].cpu()  # trapezoid / Simpson nodes (the same points)
        x_g = rules[2][i][0].cpu()
        f_g = tensors.integrands(x_g, kink, alpha).numpy()
        for f in range(3):
            nodes_t[f].set_data(x_t.numpy(), np.full(len(x_t), -0.06 * fs[f].max()))
            nodes_g[f].set_data(x_g.numpy(), f_g[f])
            for m in range(3):
                trails[3 * f + m].set_data(n[: i + 1], err[m, : i + 1, f])
        title.set_text(f"N = {ns[i]}:  blue ticks = trapezoid/Simpson points,  red dots = Gauss–Legendre nodes")

    fig.tight_layout()
    frames = frame_indices(len(ns), len(ns))
    return save_gif(fig, update, frames + frames[-1:] * 6, path, fps=4)
