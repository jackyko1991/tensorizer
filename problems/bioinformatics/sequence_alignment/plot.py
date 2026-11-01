"""Animate the DP table of pair 0 filling one anti-diagonal per frame, then the optimal path being traced back."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(H: torch.Tensor, path: list, a: str, c: str, top: str, bottom: str, scores: torch.Tensor,
            done_at: torch.Tensor, local: bool, out: Path) -> Path:  # fmt: skip
    """H (n+1, m+1) table of pair 0; path [(i, j), …]; scores (B,) and the diagonal at which each pair is done."""
    table = H.cpu().numpy()
    n1, m1 = table.shape
    diag = np.add.outer(np.arange(n1), np.arange(m1))  # i + j of every cell
    ends = done_at.cpu().numpy()  # anti-diagonal on which each pair's table is complete
    n_diag = int(ends.max()) + 1  # the batch is done when its largest table is
    n_path = max(2, len(path) // 2)
    total = n_diag + n_path + 6  # fill, trace back, hold

    fig = plt.figure(figsize=(9, 4.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 0.9])
    ax = fig.add_subplot(grid[:, 0])
    ax_s = fig.add_subplot(grid[0, 1])
    ax_t = fig.add_subplot(grid[1, 1])

    vmax = max(1.0, float(np.abs(table[np.isfinite(table)]).max()))
    cmap = plt.get_cmap("Blues" if local else "RdBu_r", 12).with_extremes(bad="#f3f4f6")  # unfilled cells: grey
    image = ax.imshow(np.ma.masked_all(table.shape), cmap=cmap, vmin=0 if local else -vmax, vmax=vmax)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="score $H_{ij}$")
    ax.grid(False)
    ax.set_xlabel("sequence c  →  j")
    ax.set_ylabel("sequence a  →  i")
    (front,) = ax.plot([], [], color="#f59e0b", lw=2)
    (trace,) = ax.plot([], [], color="#dc2626", lw=2)
    title = ax.set_title(" ", fontsize=10)  # placeholder so tight_layout leaves room

    b = np.arange(len(scores))
    bars = ax_s.bar(b, scores.cpu().numpy(), color="#93c5fd", edgecolor="#1e40af")
    ax_s.set_xlabel("pair $b$")
    ax_s.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_s.set_ylabel("best score")
    ax_s.set_title("one tensor update per diagonal, all pairs at once", fontsize=9)
    ax_s.axhline(0, color="#9ca3af", lw=0.8)
    lo, hi = min(0.0, float(scores.min())), float(scores.max())
    ax_s.set_ylim(lo - 0.1 * (hi - lo + 1), hi + 0.1 * (hi - lo + 1))

    ax_t.axis("off")
    width = 40
    marks = "".join("|" if x == y else ("·" if "-" not in (x, y) else " ") for x, y in zip(top, bottom))
    lines = []
    for k in range(0, len(top), width):
        lines += [top[k : k + width], marks[k : k + width], bottom[k : k + width], ""]
    text = ax_t.text(0, 1, "", va="top", family="monospace", fontsize=7.5, transform=ax_t.transAxes)

    def update(f: int):
        d = min(f, n_diag - 1)
        ax.images[0].set_data(np.ma.masked_where(diag > d, table))
        if f < n_diag:
            ii = np.arange(max(0, d - m1 + 1), min(n1 - 1, d) + 1)
            front.set_data(d - ii, ii)
            title.set_text(f"pair 0, anti-diagonal i + j = {d}  (batch: {d} of {n_diag - 1})")
        else:
            front.set_data([], [])
            k = min(len(path), round((f - n_diag + 1) * len(path) / n_path))
            pts = path[::-1][:k]
            trace.set_data([p[1] for p in pts], [p[0] for p in pts])
            title.set_text(f"pair 0: traceback of the optimal alignment, score {scores[0]:.0f}")
        for bar, e in zip(bars, ends):
            bar.set_alpha(1.0 if d >= e else 0.15)
        text.set_text("\n".join(lines[:16]) if f >= n_diag + n_path else "")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(total, 75), out, fps=10)
