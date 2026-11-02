"""Animate the space-time diagram of the current champion next to the halted fraction against the step budget."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

from problems._common.plotting import frame_indices, save_gif

# 0 blank cell, 1 a one, 2 head on a 0, 3 head on a 1, 4 after the machine halted
CMAP = ListedColormap(["#ffffff", "#2563eb", "#fcd34d", "#b45309", "#f3f4f6"])


def picture(rows: torch.Tensor, heads: torch.Tensor, n_rows: int, lo: int, hi: int) -> np.ndarray:
    """One space-time image (n_rows, hi − lo + 1): row t = the tape after t steps, cells lo … hi around the start."""
    start = rows.shape[1] // 2
    img = np.full((n_rows, hi - lo + 1), 4)
    cols = np.arange(lo, hi + 1) + start
    ok = (cols >= 0) & (cols < rows.shape[1])
    tape = rows.cpu().numpy()
    img[: len(tape), ok] = tape[:, cols[ok]]
    for t, h in enumerate(heads.tolist()):
        if lo <= h - start <= hi:
            img[t, h - start - lo] += 2
    return img


def extent(records: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[int, int]:
    """Smallest cell range (relative to the start) holding every 1 and head position of the given runs."""
    lo, hi = 0, 0
    for rows, heads in records:
        start = rows.shape[1] // 2
        used = np.nonzero(rows.cpu().numpy().any(axis=0))[0].tolist() + heads.tolist()
        lo, hi = min(lo, min(used) - start), max(hi, max(used) - start)
    return lo - 1, hi + 1


def animate(
    records: dict[int, tuple[torch.Tensor, torch.Tensor]],
    frac: torch.Tensor,
    frac2: torch.Tensor,
    bb4: tuple[torch.Tensor, torch.Tensor],
    labels: dict[str, str],
    path: Path,
) -> Path:
    """records: halting step s → (rows, heads) of the champion halting at s; frac (T+1,) halted fraction of the
    chosen n-state batch, frac2 the same for all 2-state machines; bb4 the space-time run of the BB(4) champion."""
    budget = len(frac) - 1
    s_max = max(records)
    lo, hi = extent(list(records.values()))
    images = {s: picture(r, h, s_max + 1, lo, hi) for s, (r, h) in records.items()}
    blank = np.full((s_max + 1, hi - lo + 1), 4)
    lo4, hi4 = extent([bb4])
    bb4_img = picture(*bb4, len(bb4[0]), lo4, hi4)

    sweep = np.unique(np.round(np.geomspace(1, budget, 90)).astype(int))  # budgets t, log-spaced
    sweep = sweep[frame_indices(len(sweep), 84)]
    tail = np.linspace(1, len(bb4[0]), 30).round().astype(int)  # then the BB(4) champion row by row
    frames = [("sweep", int(t)) for t in sweep] + [("bb4", int(k)) for k in tail]
    frames = [frames[i] for i in frame_indices(len(frames), 120)]

    fig, (ax_d, ax_f) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    image = ax_d.imshow(blank, cmap=CMAP, vmin=-0.5, vmax=4.5, interpolation="nearest", aspect="auto")
    ax_d.set_xticks([])
    ax_d.set_xlabel("tape cells (blue = 1, amber = head)")
    ax_d.set_ylabel("step (time runs down)")
    ax_d.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_d.grid(False)
    title = ax_d.set_title("champion within 100 steps: 21 steps", fontsize=10)  # placeholder sized for the layout

    t_axis = np.arange(budget + 1)
    f, f2 = frac.cpu().numpy(), frac2.cpu().numpy()
    ax_f.plot(t_axis[1:], f2[1:], color="#9ca3af", lw=1.2, label="all 2-state machines")
    ax_f.plot(t_axis[1:], f[1:], color="#2563eb", lw=1.8, label=labels["batch"])
    shade = [ax_f.fill_between(t_axis[1:2], f[1:2], 1, color="#dc2626", alpha=0.12)]
    cursor = ax_f.axvline(1, color="#f59e0b", lw=1.2)
    ax_f.set_xscale("log")
    ax_f.set_xlim(1, budget)
    ax_f.set_ylim(0, 1.02)
    ax_f.set_xlabel("step budget $t$")
    ax_f.set_ylabel("fraction halted within $t$ steps")
    ax_f.legend(loc="lower right", fontsize=8)
    status = ax_f.text(0.03, 0.97, "", transform=ax_f.transAxes, va="top", fontsize=9, family="monospace",
                       bbox={"facecolor": "white", "edgecolor": "#e5e7eb", "alpha": 0.9})  # fmt: skip

    def update(frame: tuple[str, int]):
        kind, t = frame
        if kind == "sweep":
            done = [s for s in records if s <= t]
            image.set_data(images[max(done)] if done else blank)
            image.set_extent((-0.5, hi - lo + 0.5, s_max + 0.5, -0.5))
            title.set_text(f"champion within {t} steps: {max(done) if done else '-'} steps")
            shade[0].remove()
            shade[0] = ax_f.fill_between(t_axis[1 : t + 1], f[1 : t + 1], 1, color="#dc2626", alpha=0.12, lw=0)
            cursor.set_xdata([t, t])
            lines = [f"t = {t}: halted {f[t]:.1%}", f"undecided {1 - f[t]:.1%} (shaded)"]
            if t >= s_max:
                lines.append(labels["check"])
        else:
            img = bb4_img.copy()
            img[t:] = 4
            image.set_data(img)
            image.set_extent((-0.5, hi4 - lo4 + 0.5, len(img) - 0.5, -0.5))
            title.set_text(f"BB(4) champion alone: step {t - 1}")
            lines = [f"t = {budget}: halted {f[-1]:.1%}", f"undecided {1 - f[-1]:.1%} (shaded)", labels["check"]]
            if t == len(bb4[0]):
                lines.append(labels["bb4"])
        status.set_text("\n".join(lines))

    fig.tight_layout()
    return save_gif(fig, update, frames, path, fps=10)
