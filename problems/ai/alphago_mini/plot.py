"""Replay the game: the board, the search's visit counts for each move, and its win estimate."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(boards: torch.Tensor, moves: torch.Tensor, counts: torch.Tensor, values: torch.Tensor, n: int, path: Path):
    """boards (M+1, P), moves (M,), counts (M, P+1), values (M,) for black."""
    bd, mv, ct = boards.cpu().numpy(), moves.cpu().numpy(), counts.cpu().numpy()
    win = (values.cpu().numpy() + 1) / 2
    n_moves = len(mv)

    fig, (ax_b, ax_v, ax_w) = plt.subplots(1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax in (ax_b, ax_v):
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(n - 0.4, -0.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    ax_b.set_facecolor("#e8c27a")
    for k in range(n):
        ax_b.plot([0, n - 1], [k, k], color="#5b4636", lw=1)
        ax_b.plot([k, k], [0, n - 1], color="#5b4636", lw=1)
    stones = ax_b.scatter([], [], s=1400 / n * 3, edgecolors="#111827", linewidths=1)
    last = ax_b.scatter([], [], s=60, color="#dc2626")
    title_b = ax_b.set_title("", fontsize=10)

    heat = ax_v.imshow(np.zeros((n, n)), cmap="Blues", vmin=0, vmax=1)
    title_v = ax_v.set_title("", fontsize=10)

    ax_w.plot(np.arange(1, n_moves + 1), win, color="#9ca3af", lw=1)
    (dot,) = ax_w.plot([], [], "o-", color="#2563eb", lw=2)
    ax_w.axhline(0.5, color="#374151", lw=0.8, ls="--")
    ax_w.set_xlim(0.5, max(n_moves, 2) + 0.5)
    ax_w.set_ylim(0, 1)
    ax_w.set_xlabel("move")
    ax_w.set_ylabel("P(black wins), search estimate")
    ax_w.set_title("value of the root position", fontsize=10)
    ax_w.grid(True, alpha=0.2)

    def update(i: int):
        b = bd[i + 1].reshape(n, n)
        r, c = np.nonzero(b)
        stones.set_offsets(np.c_[c, r] if len(r) else np.empty((0, 2)))
        stones.set_facecolors(["#111827" if b[y, x] > 0 else "#ffffff" for y, x in zip(r, c)])
        m = mv[i]
        last.set_offsets([[m % n, m // n]] if m < n * n else np.empty((0, 2)))
        who = "black" if i % 2 == 0 else "white"
        title_b.set_text(f"move {i + 1}: {who} {'passes' if m == n * n else 'plays'}")
        visits = ct[i][: n * n].reshape(n, n)
        heat.set_data(visits / max(visits.max(), 1))
        title_v.set_text(f"visit counts N_a ({int(ct[i].sum())} sims)")
        dot.set_data(np.arange(1, i + 2), win[: i + 1])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(n_moves, n_moves), path, fps=2)
