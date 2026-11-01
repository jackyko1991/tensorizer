"""Animate k-means in PC space next to the true cell types, the inertia and the explained variance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

BLUE, GREY = "#2563eb", "#9ca3af"


def match_colours(final: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """Colour index for each cluster: the true type most of its cells belong to (so colours line up)."""
    return np.array([np.bincount(truth[final == j]).argmax() if (final == j).any() else j for j in range(k)])


def animate(
    y: torch.Tensor,
    emb: torch.Tensor,
    labels: torch.Tensor,
    km: dict,
    ari: list[float],
    ratio: torch.Tensor,
    n_pcs: int,
    embedding: str,
    path: Path,
) -> Path:
    pc = y[:, :2].numpy()
    e = emb.numpy()
    truth = labels.numpy()
    centres = km["centres"][:, :, :2].cpu().numpy()  # (T+1, K, 2)
    assign = km["labels"].cpu().numpy()  # (T+1, C)
    inertia = km["inertia"].cpu().numpy()  # (T+1, R)
    T1, k = centres.shape[:2]
    cmap = np.array(ACCENT)
    colour_of = cmap[match_colours(assign[-1], truth, k) % len(cmap)]  # (K,) colour per cluster

    fig = plt.figure(figsize=(9.6, 4.0))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.1, 1.0])
    ax = fig.add_subplot(grid[:, 0])
    ax_t = fig.add_subplot(grid[:, 1])
    ax_i = fig.add_subplot(grid[0, 2])
    ax_v = fig.add_subplot(grid[1, 2])

    # --- k-means in PC1–PC2 ---
    dots = ax.scatter(pc[:, 0], pc[:, 1], s=6, c=colour_of[assign[0]], lw=0, alpha=0.8)
    stars = ax.scatter(centres[0, :, 0], centres[0, :, 1], s=160, marker="X", c=colour_of, edgecolors="#111827", lw=1.2)
    trails = [ax.plot([], [], color="#111827", lw=1, alpha=0.6)[0] for _ in range(k)]
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"k-means on {n_pcs} PCs (shown: PC1–PC2)", fontsize=10)
    counter = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=9, va="top")

    # --- the true cell types ---
    for t in np.unique(truth):
        sel = truth == t
        ax_t.scatter(e[sel, 0], e[sel, 1], s=6, color=cmap[t % len(cmap)], lw=0, alpha=0.8, label=f"type {t + 1}")
    ax_t.legend(loc="best", fontsize=7, markerscale=2.5, framealpha=0.9)
    name = "t-SNE" if embedding == "tsne" else "PC1–PC2"
    ax_t.set_title(f"true cell types ({name})", fontsize=10)
    ax_t.set_xticks([])
    ax_t.set_yticks([])

    # --- inertia per iteration: all restarts (grey), the kept one (blue) ---
    best = km["best"]
    its = np.arange(T1)
    for r in range(inertia.shape[1]):
        if r != best:
            ax_i.plot(its, inertia[:, r], color=GREY, lw=0.8, alpha=0.7)
    (line_i,) = ax_i.plot([], [], "o-", color=BLUE, ms=3, lw=1.6)
    ax_i.set_xlim(0, T1 - 1)
    ax_i.set_title("inertia (within-cluster SS)", fontsize=9)
    ax_i.tick_params(labelsize=7)

    # --- scree plot ---
    n_bar = min(10, len(ratio))
    ax_v.bar(np.arange(1, n_bar + 1), 100 * ratio[:n_bar].numpy(), color=[BLUE if p < n_pcs else GREY
             for p in range(n_bar)])  # fmt: skip
    ax_v.set_title("explained variance per PC (%)", fontsize=9)
    ax_v.set_xticks(np.arange(1, n_bar + 1))
    ax_v.tick_params(labelsize=7)
    ax_v.set_xlabel("component", fontsize=8)

    def update(t: int):
        dots.set_color(colour_of[assign[t]])
        stars.set_offsets(centres[t])
        for j, trail in enumerate(trails):
            trail.set_data(centres[: t + 1, j, 0], centres[: t + 1, j, 1])
        line_i.set_data(its[: t + 1], inertia[: t + 1, best])
        counter.set_text(f"iteration {t}   ARI = {ari[t]:.3f}")

    fig.tight_layout()
    steps_shown = frame_indices(T1, 40)
    frames = [t for t in steps_shown for _ in range(4)]  # hold each iteration for a few frames
    return save_gif(fig, update, frames, path, fps=8)
