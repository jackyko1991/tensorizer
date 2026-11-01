"""Animate board 0 next to the population curves of every board."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from problems._common.plotting import frame_indices, save_gif

CMAP = ListedColormap(["#ffffff", "#2563eb"])


def animate(history: torch.Tensor, pop: torch.Tensor, pattern: str, path: Path) -> Path:
    """history (G+1, B, H, W) uint8, pop (G+1, B)."""
    boards, counts = history[:, 0].cpu().numpy(), pop.cpu().numpy()
    gens = np.arange(len(counts))

    fig, (ax_b, ax_p) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    image = ax_b.imshow(boards[0], cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for side in ax_b.spines.values():
        side.set_color("#9ca3af")
    title = ax_b.set_title("", fontsize=10)

    for b in range(counts.shape[1]):
        ax_p.plot(
            gens,
            counts[:, b],
            lw=1.2 if b == 0 else 0.8,
            color="#2563eb" if b == 0 else "#9ca3af",
            alpha=1 if b == 0 else 0.6,
        )
    (dot,) = ax_p.plot([], [], "o", color="#2563eb", ms=6)
    ax_p.set_xlabel("generation")
    ax_p.set_ylabel("live cells")
    ax_p.set_title("population (blue: the board on the left)", fontsize=10)
    ax_p.grid(True, alpha=0.2)

    def update(g: int):
        image.set_data(boards[g])
        title.set_text(f"{pattern.replace('_', ' ')} — generation {g}")
        dot.set_data([g], [counts[g, 0]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(gens), 120), path, fps=12)
