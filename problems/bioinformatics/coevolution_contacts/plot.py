"""Animate contact prediction: the alignment, the top-L predicted contacts on the true map, and precision."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from problems._common.plotting import frame_indices, save_gif

from .equation import top_pairs

BLUE, RED, ORANGE, GREY = "#2563eb", "#dc2626", "#ea580c", "#9ca3af"


def animate(
    seqs: torch.Tensor,
    q: int,
    contacts: torch.Tensor,
    main_scores: torch.Tensor,
    cov_scores: torch.Tensor,
    xs: list[int],
    prec_main: list[float],
    prec_cov: list[float],
    random_level: float,
    label: str,
    depth_sweep: bool,
    min_sep: int,
    path: Path,
) -> Path:
    L = contacts.shape[0]
    n_rows = min(48, seqs.shape[0])
    letters = ListedColormap(plt.get_cmap("tab20")(np.arange(q) % 20)).with_extremes(bad="white")  # bad = unused rows
    msa = seqs[:n_rows].numpy().astype(float)  # (rows, L)

    fig = plt.figure(figsize=(9.6, 3.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25])
    ax_msa, ax_map, ax_p = (fig.add_subplot(grid[0, k]) for k in range(3))

    # --- the alignment: one row per sequence, one colour per letter ---
    msa_img = ax_msa.imshow(msa, cmap=letters, vmin=-0.5, vmax=q - 0.5, aspect="auto",
                            interpolation="nearest")  # fmt: skip
    ax_msa.set_xlabel("column i")
    ax_msa.set_ylabel("sequence n")
    ax_msa.grid(False)
    ax_msa.set_title(f"MSA (first {n_rows} rows)", fontsize=10)
    col_marks = [ax_msa.axvline(-5, color="#f59e0b", lw=2) for _ in range(2)]

    # --- contact map: true contacts (grey), predictions upper = DCA/PLM, lower = covariance ---
    ci, cj = np.nonzero(contacts.numpy())
    ax_map.scatter(cj, ci, s=22, marker="s", color="#d1d5db", lw=0, label="true contact")
    ax_map.plot([0, L - 1], [0, L - 1], color=GREY, lw=0.8)
    ax_map.set_xlim(-0.5, L - 0.5)
    ax_map.set_ylim(L - 0.5, -0.5)
    ax_map.set_aspect("equal")
    ax_map.set_title(f"top-L: {label} (upper)\ncovariance (lower)", fontsize=9)
    ax_map.tick_params(labelsize=8)
    hit_up = ax_map.scatter([], [], s=10, color=BLUE, label="correct")
    miss_up = ax_map.scatter([], [], s=10, color=RED, label="wrong")
    hit_lo = ax_map.scatter([], [], s=10, color=BLUE)
    miss_lo = ax_map.scatter([], [], s=10, color=RED)
    ax_map.legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    # --- precision of the top-L pairs ---
    ax_p.set_ylim(0, 1.05)
    ax_p.set_ylabel("top-L precision")
    if depth_sweep:
        ax_p.set_xscale("log")
        ticks = [t for t in (20, 50, 100, 200, 500, 1000, 2000, 4000) if xs[0] <= t <= xs[-1]]
        ax_p.set_xticks(ticks, [str(t) for t in ticks])
        ax_p.minorticks_off()
        ax_p.set_xlabel("sequences in the MSA, N")
    else:
        ax_p.set_xlabel("Adam iteration")
    ax_p.set_xlim(xs[0], xs[-1])
    ax_p.axhline(random_level, color=GREY, ls="--", lw=1)
    ax_p.text(xs[-1], random_level + 0.02, "random", ha="right", fontsize=8, color="#6b7280")
    (line_main,) = ax_p.plot([], [], "o-", color=BLUE, ms=3, lw=1.6, label=label)
    (line_cov,) = ax_p.plot([], [], "s-" if depth_sweep else "-", color=ORANGE, ms=3, lw=1.4, label="covariance")
    ax_p.legend(loc="center right", fontsize=8)
    ax_p.set_title("precision vs true contacts", fontsize=10)
    counter = ax_p.text(0.03, 0.1, "", transform=ax_p.transAxes, fontsize=9)

    def split(pairs: torch.Tensor, upper: bool):
        i, j = pairs[:, 0].numpy(), pairs[:, 1].numpy()
        ok = contacts[pairs[:, 0], pairs[:, 1]].numpy()
        xy = np.stack([j, i], 1) if upper else np.stack([i, j], 1)  # (column, row) for the scatter
        return xy[ok], xy[~ok]

    def update(s: int):
        top = top_pairs(main_scores[s], L, min_sep)
        good, bad = split(top, True)
        hit_up.set_offsets(good.reshape(-1, 2))
        miss_up.set_offsets(bad.reshape(-1, 2))
        good, bad = split(top_pairs(cov_scores[s], L, min_sep), False)
        hit_lo.set_offsets(good.reshape(-1, 2))
        miss_lo.set_offsets(bad.reshape(-1, 2))
        for mark, col in zip(col_marks, top[0].tolist()):  # the strongest predicted pair, in the MSA
            mark.set_xdata([col, col])
        if depth_sweep:
            shown = msa.copy()
            shown[xs[s] :] = np.nan  # only the first N sequences are used in this frame
            msa_img.set_data(shown)
            counter.set_text(f"N = {xs[s]}")
        else:
            counter.set_text(f"iteration {xs[s] + 1}")
        line_main.set_data(xs[: s + 1], prec_main[: s + 1])
        line_cov.set_data(xs[: s + 1], prec_cov[: s + 1])

    fig.tight_layout()
    frames = frame_indices(len(xs), 60)
    frames = frames + [frames[-1]] * (len(frames) // 3)  # hold the final picture
    return save_gif(fig, update, frames, path, fps=8)
