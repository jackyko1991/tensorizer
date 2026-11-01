"""Left: the tanks draining through holes of different sizes. Right: h(t) against the exact solution."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon, Rectangle

from problems._common.plotting import ACCENT, frame_indices, save_gif

JET_LENGTH = 0.35  # how far below the tank the falling jet is drawn (in units of h0)
WIDEN = 6.0  # holes and jets are drawn this many times wider than in reality


def jet_outline(x0: float, d: float, v0: float, h0: float, g: float) -> np.ndarray:
    """Outline of the falling jet: it narrows as it speeds up (continuity: width² × speed = const)."""
    depth = np.linspace(0, JET_LENGTH * h0, 20)
    speed = np.sqrt(v0**2 + 2 * g * depth)
    half = 0.5 * WIDEN * d * np.sqrt(v0 / speed)
    return np.concatenate([np.column_stack([x0 - half, -depth]), np.column_stack([x0 + half, -depth])[::-1]])


def animate(
    times: torch.Tensor, levels: torch.Tensor, exact: torch.Tensor, d: torch.Tensor, D: float, h0: float, g: float,
    path: Path,
) -> Path:  # fmt: skip
    t = times.cpu().numpy()  # (T,)
    h = levels.cpu().numpy()  # (T, B)
    he = exact.cpu().numpy()  # (T, B)
    diam = d.cpu().numpy()  # (B,)
    n = len(diam)
    gap = 1.5 * D
    colors = [ACCENT[b % len(ACCENT)] for b in range(n)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax.set_xlim(-0.2 * D, n * gap)
    ax.set_ylim(-JET_LENGTH * h0 * 1.05, h0 * 1.25)
    ax.set_xlabel("tanks (hole diameter)")
    ax.set_ylabel("h (m)")
    ax.set_xticks([(b + 0.4) * gap for b in range(n)], [f"{100 * x:.1f} cm" for x in diam], fontsize=8)
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)  # same colour as the tank's curve on the right
    ax.grid(False)
    water, jets = [], []
    for b in range(n):
        x0 = b * gap
        ax.plot([x0, x0, x0 + D, x0 + D], [h0 * 1.15, 0, 0, h0 * 1.15], color="#374151", lw=2)
        water.append(ax.add_patch(Rectangle((x0, 0), D, h[0, b], fc="#93c5fd", ec="none")))
        jets.append(ax.add_patch(Polygon(np.zeros((2, 2)), fc="#60a5fa", ec="none")))
    clock = ax.set_title(" ")

    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(0, h0 * 1.05)
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("h (m)")
    for b in range(n):
        ax2.plot(t, he[:, b], color="#374151", lw=0.8, ls="--")
    curves = [ax2.plot([], [], color=colors[b], lw=2)[0] for b in range(n)]
    ax2.set_title("dashed: exact (√h₀ − (a/A)√(g/2) t)²", fontsize=9)

    def update(step: int):
        for b in range(n):
            water[b].set_height(h[step, b])
            x_hole = b * gap + D / 2
            v0 = np.sqrt(2 * g * h[step, b])
            jets[b].set_xy(jet_outline(x_hole, diam[b], v0, h0, g) if v0 > 1e-3 else np.zeros((2, 2)))
            curves[b].set_data(t[: step + 1], h[: step + 1, b])
        clock.set_text(f"t = {t[step]:.0f} s   (holes drawn {WIDEN:.0f}× wider)")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path, fps=16)
