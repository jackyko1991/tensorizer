"""Animate training: the most important filter turning into the planted motif (as a logo), loss and accuracy."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath

from problems._common.plotting import frame_indices, save_gif

from .equation import information

BLUE, ORANGE, GREY = "#2563eb", "#ea580c", "#9ca3af"
BASE_COLOURS = {"A": "#16a34a", "C": "#2563eb", "G": "#f59e0b", "T": "#dc2626"}
FONT = FontProperties(family="DejaVu Sans", weight="bold")


def letter(ax, ch: str, x: float, y: float, height: float) -> PathPatch:
    """Draw base `ch` stretched to fill the box [x − 0.45, x + 0.45] × [y, y + height]."""
    path = TextPath((0, 0), ch, size=1, prop=FONT)
    bb = path.get_extents()
    t = mtransforms.Affine2D().translate(-bb.x0, -bb.y0).scale(0.9 / bb.width, height / bb.height)
    t = t.translate(x - 0.45, y)
    return ax.add_patch(PathPatch(t.transform_path(path), color=BASE_COLOURS[ch], lw=0))


def draw_logo(ax, pwm: np.ndarray, x0: int = 0) -> list[PathPatch]:
    """Sequence logo: at each position the letters are stacked, height = p_kb · IC_k (bits), largest on top."""
    ic = information(torch.from_numpy(pwm)).numpy()
    patches = []
    for k in range(pwm.shape[0]):
        y = 0.0
        for b in np.argsort(pwm[k]):  # smallest first, so the dominant base ends on top
            h = pwm[k, b] * ic[k]
            if h > 0.01:
                patches.append(letter(ax, "ACGT"[b], x0 + k, y, h))
            y += h
    return patches


def setup_logo_axis(ax, width: int, title: str) -> None:
    ax.set_xlim(-0.6, width - 0.4)
    ax.set_ylim(0, 2.05)
    ax.set_ylabel("bits", fontsize=8)
    ax.set_xticks(range(width), [str(k + 1) for k in range(width)], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(False)
    ax.set_title(title, fontsize=10)


def animate(
    motif: torch.Tensor,
    consensus: str,
    pwm_snaps: torch.Tensor,
    r_snaps: list[float],
    hist: dict,
    oracle: float,
    best: int,
    shift: int,
    path: Path,
) -> Path:
    w = pwm_snaps.shape[1]
    loss = hist["loss"].numpy()
    n_steps = len(loss)
    kernel = max(1, n_steps // 50)
    csum = np.concatenate([[0.0], np.cumsum(loss)])
    lo = np.maximum(np.arange(n_steps) + 1 - kernel, 0)
    smooth = (csum[1:] - csum[lo]) / (np.arange(n_steps) + 1 - lo)  # running mean over the last `kernel` batches
    it_smooth = np.arange(1, n_steps + 1)  # step numbers 1, 2, … (log axis below)
    ev = np.array(hist["eval_steps"]) + 1
    snaps = np.array(hist["snap_steps"]) + 1

    fig = plt.figure(figsize=(9.2, 4.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax_m = fig.add_subplot(grid[0, 0])
    ax_f = fig.add_subplot(grid[1, 0])
    ax_l = fig.add_subplot(grid[0, 1])
    ax_a = fig.add_subplot(grid[1, 1])

    # --- planted motif (top) and the learned filter (bottom), both as logos ---
    setup_logo_axis(ax_m, w, f"planted motif {consensus}")
    draw_logo(ax_m, motif.numpy(), shift)  # drawn where the trained filter holds it
    setup_logo_axis(ax_f, w, "")
    ax_f.set_xlabel(f"position in the filter (width {w})", fontsize=8)
    logo_patches: list[PathPatch] = []

    # --- loss ---
    ax_l.plot(it_smooth, smooth, color=GREY, lw=1, alpha=0.5)
    (line_l,) = ax_l.plot([], [], color=BLUE, lw=1.6, label="train (batch avg)")
    (line_lt,) = ax_l.plot([], [], "o-", color=ORANGE, ms=2.5, lw=1.2, label="test")
    ax_l.set_xscale("log")
    ax_l.set_xlim(1, n_steps)
    ax_l.set_ylim(0, max(0.75, smooth.max() * 1.05))
    ax_l.set_title("cross-entropy loss", fontsize=10)
    ax_l.legend(loc="upper right", fontsize=7)
    ax_l.tick_params(labelsize=7)

    # --- accuracy ---
    ax_a.axhline(oracle, color=GREY, ls="--", lw=1)
    ax_a.text(1.2, oracle - 0.015, "true-PWM scan", ha="left", va="top", fontsize=7, color="#6b7280")
    (line_a,) = ax_a.plot([], [], "o-", color=BLUE, ms=2.5, lw=1.4, label="train")
    (line_at,) = ax_a.plot([], [], "o-", color=ORANGE, ms=2.5, lw=1.4, label="test")
    ax_a.set_xscale("log")
    ax_a.set_xlim(1, n_steps)
    ax_a.set_ylim(0.4, 1.0)
    ax_a.set_xlabel("Adam step (mini-batch, log scale)", fontsize=8)
    ax_a.set_title("accuracy", fontsize=10)
    ax_a.legend(loc="lower right", fontsize=7)
    ax_a.tick_params(labelsize=7)

    def update(s: int):
        for p in logo_patches:
            p.remove()
        logo_patches[:] = draw_logo(ax_f, pwm_snaps[s].numpy())
        ax_f.set_title(f"filter #{best + 1} read as a PWM, step {snaps[s]}:  r = {r_snaps[s]:.2f}", fontsize=10)
        upto = it_smooth <= snaps[s]
        line_l.set_data(it_smooth[upto], smooth[upto])
        e = ev <= snaps[s]
        line_lt.set_data(ev[e], np.array(hist["test_loss"])[e])
        line_a.set_data(ev[e], np.array(hist["train_acc"])[e])
        line_at.set_data(ev[e], np.array(hist["test_acc"])[e])

    fig.tight_layout()
    frames = frame_indices(len(snaps), 40)  # snapshots are log-spaced in steps
    frames = frames + [frames[-1]] * 8  # hold the final logo
    return save_gif(fig, update, frames, path, fps=6)
