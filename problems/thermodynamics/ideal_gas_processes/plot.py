"""Animate a state point moving along each process (area under it shaded) beside running W and Q bars."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(paths: torch.Tensor, r: dict, names: list[str], path: Path) -> Path:
    V = 1e3 * paths[..., 0].numpy()  # (4, K) litres
    p = 1e-3 * paths[..., 1].numpy()  # (4, K) kPa
    W_run, Q_run = r["W_run"].numpy(), r["Q_run"].numpy()  # (4, K) joules
    n_proc, K = V.shape

    fig, (ax_pv, ax_e) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: p–V diagram. Each area patch is the region under the path traced so far: its work W.
    areas, dots = [], []
    for c in range(n_proc):
        ax_pv.plot(V[c], p[c], color=ACCENT[c], lw=2, label=names[c])
        area = Polygon(np.zeros((1, 2)), closed=True, color=ACCENT[c], alpha=0.15, lw=0)
        areas.append(ax_pv.add_patch(area))
        dots.append(ax_pv.plot([], [], "o", color=ACCENT[c])[0])
    ax_pv.set_xlim(0, V.max() * 1.05)
    ax_pv.set_ylim(0, p.max() * 1.1)
    ax_pv.set_xlabel("V (L)")
    ax_pv.set_ylabel("p (kPa)")
    ax_pv.set_title("shaded area = W = ∫ p dV")
    ax_pv.legend(fontsize=8, loc="upper right")

    # Right: running totals of work done by the gas and heat added (Q = ΔU + W).
    x = np.arange(n_proc)
    bars_W = ax_e.bar(x - 0.2, np.zeros(n_proc), width=0.4, color=ACCENT[:n_proc], label="W")
    bars_Q = ax_e.bar(x + 0.2, np.zeros(n_proc), width=0.4, color=ACCENT[:n_proc], alpha=0.4, hatch="//", label="Q")
    low, high = min(0.0, Q_run.min(), W_run.min()), max(Q_run.max(), W_run.max())
    ax_e.set_ylim(low - 0.1 * (high - low), high + 0.1 * (high - low))
    ax_e.axhline(0, color="black", lw=0.8)
    ax_e.set_xticks(x, names, fontsize=7, rotation=20)
    ax_e.set_ylabel("energy (J)")
    ax_e.set_title("solid: work W    hatched: heat Q", fontsize=9)

    def update(step: int):
        for c in range(n_proc):
            traced = np.column_stack([V[c, : step + 1], p[c, : step + 1]])  # path so far
            floor = np.array([[V[c, step], 0.0], [V[c, 0], 0.0]])  # down to the V axis and back
            areas[c].set_xy(np.vstack([traced, floor]))
            dots[c].set_data([V[c, step]], [p[c, step]])
            bars_W[c].set_height(W_run[c, step])
            bars_Q[c].set_height(Q_run[c, step])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(K, 80), path)
