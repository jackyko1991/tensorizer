"""Animate Baum–Welch: posterior P(island) along sequence 0 against the planted truth, one frame per iteration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from problems._common.plotting import frame_indices, save_gif

STRIP = ListedColormap(["#ffffff", "#f59e0b"])


def animate(x0, z0, post0, paths0, models, truth, log_lik, true_ll, out: Path) -> Path:
    """x0, z0 (T,) letters and true states of sequence 0; post0 (I+1, T); paths0 list of (T,) Viterbi paths;
    models list of (π, A, E); truth (π, A, E); log_lik list (I+1,); true_ll float."""
    T = len(z0)
    pos = np.arange(T)
    truth_z = z0.cpu().numpy()
    gc = x0.cpu().numpy()
    gc = ((gc == 1) | (gc == 2)).astype(float)
    window = 51
    gc_smooth = np.convolve(gc, np.ones(window) / window, mode="same")  # GC fraction in a sliding window
    post = post0.cpu().numpy()
    paths = [p.cpu().numpy() for p in paths0]
    E_true = truth[2].cpu().numpy()
    n = len(models)

    fig = plt.figure(figsize=(9, 5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[0, :])
    ax_l = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])

    ax.fill_between(pos, 0, truth_z, step="mid", color="#dbeafe", label="planted island")
    ax.plot(pos, gc_smooth, color="#9ca3af", lw=0.8, label=f"GC fraction ({window}-letter window)")
    (line,) = ax.plot(pos, post[0], color="#2563eb", lw=1.3, label="posterior P(island)")
    strip = ax.imshow(paths[0][None], cmap=STRIP, vmin=0, vmax=1, aspect="auto", extent=(-0.5, T - 0.5, -0.2, -0.07))
    ax.text(T * 0.005, -0.135, "Viterbi", fontsize=7, va="center", color="#92400e")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.22, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("position $t$ in sequence 0")
    ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.9)
    title = ax.set_title(" ", fontsize=10)  # placeholder so tight_layout leaves room

    its = np.arange(n)
    ax_l.plot(its, log_lik, color="#9ca3af", lw=1, marker="o", ms=3)
    ax_l.axhline(true_ll, color="#16a34a", ls="--", lw=1, label="true model")
    (dot,) = ax_l.plot([], [], "o", color="#2563eb", ms=7)
    ax_l.set_xlabel("Baum–Welch iteration")
    ax_l.set_ylabel("log P(x)")
    ax_l.legend(fontsize=7, loc="lower right")
    ax_l.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    letters = np.arange(4)
    bars = []
    for k, (color, name) in enumerate([("#9ca3af", "background"), ("#2563eb", "island")]):
        bars.append(ax_e.bar(letters + (k - 0.5) * 0.38, models[0][2][k].cpu().numpy(), 0.36, color=color, label=name))
        ax_e.plot(letters + (k - 0.5) * 0.38, E_true[k], "_", color="#111827", ms=12, mew=2)
    ax_e.set_xticks(letters)
    ax_e.set_xticklabels(list("ACGT"))
    ax_e.set_ylim(0, 0.5)
    ax_e.set_ylabel("emission $E_{k\\alpha}$")
    ax_e.set_title("bars: current model, black ticks: true", fontsize=9)
    ax_e.legend(fontsize=7, loc="upper left", ncol=2)

    def update(i: int):
        line.set_ydata(post[i])
        strip.set_data(paths[i][None])
        acc = (paths[i] == truth_z).mean()
        title.set_text(f"iteration {i}: Viterbi agrees with the planted islands at {acc:.1%} of positions")
        dot.set_data([i], [log_lik[i]])
        for k in range(2):
            for bar, h in zip(bars[k], models[i][2][k].tolist()):
                bar.set_height(h)

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(n, 60), out, fps=2)
