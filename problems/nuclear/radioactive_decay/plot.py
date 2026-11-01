"""Animate the grid of nuclei decaying next to the counted N(t) and the exponential law."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap

from problems._common.plotting import frame_indices, save_gif


def animate(
    times: torch.Tensor, history: torch.Tensor, counts: torch.Tensor, exact: torch.Tensor, half_life: float, path: Path
) -> Path:
    t = times.cpu().numpy()
    grid = history.cpu().numpy().astype(int)  # (T, S, S) 1 = undecayed
    n_sim = counts.cpu().numpy()
    n_exact = exact.cpu().numpy()
    n0 = n_exact[0]

    fig, (ax_grid, ax_n) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1, 1.3]})
    cmap = ListedColormap(["#e5e7eb", "#2563eb"])  # decayed, undecayed
    image = ax_grid.imshow(grid[0], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.grid(False)
    title = ax_grid.set_title("")

    ax_n.plot(t, n_exact, color="#9ca3af", lw=2, ls="--", label=r"$N_0 e^{-\lambda t}$")
    (line,) = ax_n.plot([], [], color="#2563eb", lw=1.8, label="simulated nuclei")
    for k in range(1, int(t[-1] / half_life) + 1):
        ax_n.axvline(k * half_life, color="#f59e0b", lw=0.8, alpha=0.6)
        ax_n.text(k * half_life, n0 * 1.02, f"{k}T½", ha="center", fontsize=8, color="#b45309")
    ax_n.set_xlim(0, t[-1])
    ax_n.set_ylim(0, n0 * 1.1)
    ax_n.set_xlabel("t (s)")
    ax_n.set_ylabel("undecayed nuclei N")
    ax_n.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=8)

    def update(step: int):
        image.set_data(grid[step])
        line.set_data(t[: step + 1], n_sim[: step + 1])
        title.set_text(f"t = {t[step]:.1f} s   N = {int(n_sim[step])} / {int(n0)}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
