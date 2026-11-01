"""Animate a window walking round the circular chromosome: skew ring, cumulative skew curve and the k-mer spectrum."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba_array

from problems._common.plotting import frame_indices, save_gif

POS, NEG, GC, ORI, TER = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#9333ea"


def animate(
    win: dict[str, torch.Tensor],
    S: torch.Tensor,
    counts: torch.Tensor,
    names: list[str],
    L: int,
    planted: tuple[int, int],
    found: tuple[int, int],
    path: Path,
) -> Path:
    start = win["start"].cpu().numpy()
    skew, gc = win["skew"].cpu().numpy(), win["gc"].cpu().numpy()
    n_win = len(start)
    step = max(1, L // 2000)  # ~2000 points are plenty for drawing the per-base curve
    pos_kb = np.arange(0, L, step) / 1e3
    S_np = S[::step].cpu().numpy()

    fig = plt.figure(figsize=(8.4, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1.2, 1])
    ax_c = fig.add_subplot(grid[:, 0])
    ax_s = fig.add_subplot(grid[0, 1])
    ax_k = fig.add_subplot(grid[1, 1])

    # --- the circular chromosome: skew bars on the outer ring, GC deviation on the inner ring ---
    ax_c.set_aspect("equal")
    ax_c.set_xlim(-1.62, 1.62)
    ax_c.set_ylim(-1.62, 1.62)
    ax_c.axis("off")
    a0 = 2 * np.pi * start / L  # angle where each window starts (clockwise from the top)
    a1 = 2 * np.pi * np.minimum(start + start[1], L) / L
    s_scale = 0.35 / max(np.abs(skew).max(), 1e-9)
    g_scale = 0.25 / max(np.abs(gc - gc.mean()).max(), 1e-9)
    skew_ring = PolyCollection(_wedges(a0, a1, 1.05, 1.05 + skew * s_scale), lw=0)
    gc_ring = PolyCollection(_wedges(a0, a1, 0.62, 0.62 + (gc - gc.mean()) * g_scale), lw=0)
    skew_rgba = to_rgba_array(np.where(skew > 0, POS, NEG))  # (W, 4)
    gc_rgba = to_rgba_array([GC] * n_win)
    ax_c.add_collection(skew_ring)
    ax_c.add_collection(gc_ring)
    ring = np.linspace(0, 2 * np.pi, 200)
    ax_c.plot(1.05 * np.sin(ring), 1.05 * np.cos(ring), color="#374151", lw=1)
    ax_c.plot(0.62 * np.sin(ring), 0.62 * np.cos(ring), color="#9ca3af", lw=0.8)
    for q in range(4):  # position ticks every quarter of the genome
        a = q * np.pi / 2
        ax_c.text(1.58 * np.sin(a), 1.58 * np.cos(a), f"{q * L / 4e3:.0f} kb", ha="center", va="center", fontsize=8)
    for p, color, label in [(planted[0], ORI, "ori"), (planted[1], TER, "ter")]:
        a = 2 * np.pi * p / L
        ax_c.plot([1.45 * np.sin(a)], [1.45 * np.cos(a)], marker=(3, 0, 180 - np.degrees(a)), color=color, ms=7)
        ax_c.text(1.3 * np.sin(a), 1.3 * np.cos(a), label, color=color, ha="center", va="center", fontsize=8)
    ax_c.text(0, 0, "GC skew (outer)\nGC content (inner)", ha="center", va="center", fontsize=7.5, color="#6b7280")
    pointer = ax_c.plot([], [], color="#374151", lw=1.2)[0]
    ax_c.set_title("circular chromosome", fontsize=10)

    # --- cumulative skew S_n, drawn up to the window ---
    ax_s.set_xlim(0, L / 1e3)
    pad = 0.08 * (S_np.max() - S_np.min() + 1)
    ax_s.set_ylim(S_np.min() - pad, S_np.max() + pad)
    ax_s.set_xlabel("position (kb)", labelpad=1)
    ax_s.set_ylabel("cumulative G − C")
    ax_s.set_title("cumulative GC skew: min = origin, max = terminus", fontsize=10)
    for p, color in [(planted[0], ORI), (planted[1], TER)]:
        ax_s.axvline(p / 1e3, color=color, ls="--", lw=1, alpha=0.7)
    curve = ax_s.plot([], [], color="#374151", lw=1.4)[0]
    found_marks = ax_s.plot(
        [found[0] / 1e3, found[1] / 1e3], [S_np.min(), S_np.max()], "o", ms=7, mfc="none", mew=1.8, color="#111827"
    )[0]
    found_label = ax_s.text(0.98, 0.06, "", transform=ax_s.transAxes, fontsize=8, ha="right")

    # --- k-mer spectrum: one vertical line per word, coloured by its number of G/C letters ---
    c = counts.cpu().numpy()
    k = len(names[0])
    n_gc = np.array([sum(ch in "GC" for ch in w) for w in names])
    cmap = plt.get_cmap("viridis", k + 1)
    ax_k.vlines(np.arange(len(c)), 0, c, colors=cmap(n_gc), lw=max(0.5, 220 / len(c)))
    ax_k.set_xlim(-1, len(c))
    ax_k.set_ylim(0, c.max() * 1.15)
    ax_k.set_ylabel("count", labelpad=1)
    ax_k.set_xlabel(f"{k}-mer index ({names[0]} … {names[-1]})", labelpad=1)
    ax_k.set_title(f"{k}-mer spectrum, colour = number of G/C (0–{k})", fontsize=10)
    ax_k.grid(axis="x", visible=False)

    def update(i: int):
        upto = start[i] + start[1]
        shown = (np.arange(n_win) <= i)[:, None]  # windows the pointer has passed
        skew_ring.set_facecolor(np.where(shown, skew_rgba, 0.0))  # alpha 0 hides the rest
        gc_ring.set_facecolor(np.where(shown, gc_rgba, 0.0))
        a = 2 * np.pi * upto / L
        pointer.set_data([0.4 * np.sin(a), 1.45 * np.sin(a)], [0.4 * np.cos(a), 1.45 * np.cos(a)])
        m = pos_kb <= upto / 1e3
        curve.set_data(pos_kb[m], S_np[m])
        done = i == n_win - 1
        found_marks.set_visible(done)
        found_label.set_text(f"○ found: ori {found[0] / 1e3:.1f} kb, ter {found[1] / 1e3:.1f} kb" if done else "")

    fig.tight_layout()
    frames = frame_indices(n_win, 60)
    frames += [frames[-1]] * (len(frames) // 6)  # hold the final picture for a moment
    return save_gif(fig, update, frames, path, fps=15)


def _wedges(a0: np.ndarray, a1: np.ndarray, r0: float, r1: np.ndarray) -> list[np.ndarray]:
    """One quadrilateral per window between radii r0 and r1[w], angles measured clockwise from the top."""
    out = []
    for w in range(len(a0)):
        a = np.array([a0[w], a1[w], a1[w], a0[w]])
        r = np.array([r0, r0, r1[w], r1[w]])
        out.append(np.stack([r * np.sin(a), r * np.cos(a)], axis=1))
    return out
