"""Corrupted faces recalled by both networks as more faces are stored, next to the capacity curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

SHOWN = 5  # faces shown in the image grid
ROWS = ["query", "classic", "modern", "stored"]


def animate(
    faces: torch.Tensor,
    queries: torch.Tensor,
    classic: torch.Tensor,
    modern: torch.Tensor,
    loads: torch.Tensor,
    m_classic: torch.Tensor,
    m_modern: torch.Tensor,
    curve: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    path: Path,
) -> Path:
    """faces, queries (P, S, S); classic, modern (F, P, N); loads, m_classic, m_modern (F,); curve = α, classic,
    modern overlaps of random patterns."""
    s = faces.shape[-1]
    n = s * s
    img_c = classic[:, :SHOWN].reshape(len(loads), SHOWN, s, s).cpu().numpy()
    img_m = modern[:, :SHOWN].reshape(len(loads), SHOWN, s, s).cpu().numpy()
    fixed = {"query": queries[:SHOWN].cpu().numpy(), "stored": faces[:SHOWN].cpu().numpy()}
    p = loads.cpu().numpy()
    mc, mm = m_classic.cpu().numpy(), m_modern.cpu().numpy()

    fig = plt.figure(figsize=(10, 5.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.5, wspace=0.18)
    inner = grid[:, 0].subgridspec(len(ROWS), SHOWN, hspace=0.08, wspace=0.08)
    images = {}
    for r, row in enumerate(ROWS):
        for c in range(SHOWN):
            ax = fig.add_subplot(inner[r, c])
            data = fixed[row][c] if row in fixed else np.zeros((s, s))
            images[row, c] = ax.imshow(-data, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if c == 0:
                ax.set_ylabel(row, fontsize=9)
    head = fig.text(0.29, 0.95, "", ha="center", fontsize=10)

    ax_f = fig.add_subplot(grid[0, 1])
    ax_f.plot(p, mc, color="#d1d5db", lw=1)
    ax_f.plot(p, mm, color="#d1d5db", lw=1)
    (line_c,) = ax_f.plot([], [], "o-", color="#dc2626", ms=3, lw=1.5, label="classic (Hebb)")
    (line_m,) = ax_f.plot([], [], "s-", color="#2563eb", ms=3, lw=1.5, label="modern (softmax)")
    ax_f.set_ylim(0.3, 1.04)
    ax_f.set_xlim(0, p[-1] + 1)
    ax_f.set_xlabel("faces stored P")
    ax_f.set_ylabel("recall overlap m")
    ax_f.set_title("corrupted faces", fontsize=10)
    ax_f.legend(fontsize=8, frameon=False, loc="lower left")

    alphas, cap_c, cap_m = (t.numpy() for t in curve)
    ax_a = fig.add_subplot(grid[1, 1])
    ax_a.semilogx(alphas, cap_c, "o-", color="#dc2626", ms=3, lw=1.5, label="classic")
    ax_a.semilogx(alphas, cap_m, "s-", color="#2563eb", ms=3, lw=1.5, label="modern")
    ax_a.axvline(0.138, color="#f59e0b", ls="--", lw=1.2, label="α_c ≈ 0.138")
    load = ax_a.axvline(p[0] / n, color="#6b7280", lw=1, ls=":", label="faces now")
    ax_a.set_ylim(0.3, 1.04)
    ax_a.set_xlim(8e-4, 2.5)
    ax_a.set_xlabel("load α = P / N")
    ax_a.set_ylabel("recall overlap m")
    ax_a.set_title("random ±1 patterns (10 % bits flipped)", fontsize=10)
    ax_a.legend(fontsize=7, frameon=False, loc="lower left")

    def update(k: int):
        for c in range(SHOWN):
            stored = c < p[k]  # faces not stored yet stay grey
            images["classic", c].set_data(-img_c[k, c] if stored else np.zeros((s, s)))
            images["modern", c].set_data(-img_m[k, c] if stored else np.zeros((s, s)))
        line_c.set_data(p[: k + 1], mc[: k + 1])
        line_m.set_data(p[: k + 1], mm[: k + 1])
        load.set_xdata([p[k] / n, p[k] / n])
        head.set_text(f"{p[k]} faces stored in N = {n} neurons (α = {p[k] / n:.3f})")

    return save_gif(fig, update, frame_indices(len(p), 90), path, fps=4)
