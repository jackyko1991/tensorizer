"""Points flowing from one shape to the next along the plan, the plan for a sweep of ε, and the cost against ε."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from problems._common.plotting import frame_indices, save_gif

from .equation import interpolate

LINES = 70  # source points whose plan rows are drawn
TOP = 6  # largest plan entries drawn per row


def animate(legs: list[dict], eps: torch.Tensor, path: Path, per_leg: int) -> Path:
    """legs: one dict per transport (x, y, tx (N, 2), plan (E, N, N), cost, dual (E,), exact, div, name)."""
    hold = max(2, per_leg // 5)
    frames = [(k, min(1.0, s / per_leg)) for k in range(len(legs)) for s in range(per_leg + hold)]
    e = eps.numpy()
    rows = np.linspace(0, len(legs[0]["x"]) - 1, LINES).astype(int)

    fig = plt.figure(figsize=(11, 3.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3, bottom=0.15)
    ax_f = fig.add_subplot(grid[0])
    ax_p = fig.add_subplot(grid[1])
    for ax in (ax_f, ax_p):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    empty = np.zeros((0, 2))
    target = ax_f.scatter(empty[:, 0], empty[:, 1], s=6, color="#d1d5db")
    moving = ax_f.scatter(empty[:, 0], empty[:, 1], s=7, color="#2563eb")
    title_f = ax_f.set_title("", fontsize=10)
    src_p = ax_p.scatter(empty[:, 0], empty[:, 1], s=4, color="#93c5fd")
    dst_p = ax_p.scatter(empty[:, 0], empty[:, 1], s=4, color="#fca5a5")
    lines = LineCollection([], linewidths=0.6)
    ax_p.add_collection(lines)
    title_p = ax_p.set_title("", fontsize=10)

    ax_c = fig.add_subplot(grid[2])
    (l_cost,) = ax_c.semilogx([], [], "o-", color="#2563eb", ms=3, label="⟨P_ε, C⟩")
    (l_dual,) = ax_c.semilogx([], [], "s--", color="#9333ea", ms=3, label="OT_ε (dual value)")
    (l_div,) = ax_c.semilogx([], [], "^-", color="#16a34a", ms=3, label="Sinkhorn divergence S_ε")
    l_exact = ax_c.axhline(0, color="#dc2626", ls=":", lw=1.3, label="exact W₂² (Hungarian)")
    marker = ax_c.axvline(e[0], color="#f59e0b", lw=1)
    ax_c.set_xlim(e.max() * 1.4, e.min() * 0.7)  # ε decreases to the right
    ax_c.set_xlabel("regularisation ε → 0")
    ax_c.set_ylabel("transport cost")
    ax_c.legend(fontsize=7, frameon=False, loc="upper left")
    title_c = ax_c.set_title("", fontsize=9)

    def update(i: int):
        k, t = frames[i]
        leg = legs[k]
        x, y = leg["x"].numpy(), leg["y"].numpy()
        target.set_offsets(y)
        moving.set_offsets(interpolate(leg["x"], leg["tx"], t).numpy())
        title_f.set_text(f"{leg['name']}, t = {t:.2f}")
        j = int(round(t * (len(e) - 1)))  # ε sharpens while the points travel
        p = leg["plan"][j].numpy()
        src_p.set_offsets(x)
        dst_p.set_offsets(y)
        top = np.argsort(p[rows], axis=1)[:, -TOP:]  # (LINES, TOP) columns of the largest entries
        segs = np.stack([np.repeat(x[rows], TOP, axis=0), y[top.ravel()]], axis=1)  # (LINES·TOP, 2, 2)
        weight = np.take_along_axis(p[rows], top, axis=1) / p[rows].sum(1, keepdims=True)
        lines.set_segments(segs)
        lines.set_color([(0.22, 0.25, 0.32, min(1.0, w)) for w in weight.ravel()])
        title_p.set_text(f"plan P_ε (largest entries), ε = {e[j]:.3g}")
        cost, dual = leg["cost"].numpy(), leg["dual"].numpy()
        big, div = (v.numpy() for v in leg["div"])
        l_cost.set_data(e, cost)
        l_dual.set_data(e, dual)
        l_div.set_data(big, div)
        l_exact.set_ydata([leg["exact"], leg["exact"]])
        ax_c.set_ylim(min(0.0, div.min()) - 0.02, max(cost.max(), dual.max()) * 1.08)
        marker.set_xdata([e[j], e[j]])
        title_c.set_text(f"{leg['name']}: W₂² = {leg['exact']:.3f}, ⟨P,C⟩ at ε = {e[-1]:.3g} is {cost[-1]:.3f}")

    return save_gif(fig, update, frame_indices(len(frames), 90), path, fps=8)
