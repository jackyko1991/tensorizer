"""Animate the two reduced Bloch vectors (left) and entanglement entropy / concurrence vs time (right)."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif


def draw_sphere(ax, title: str) -> None:
    """Faint unit sphere with equator, a meridian and the x, y, z axes."""
    u = np.linspace(0, 2 * np.pi, 60)
    ax.plot(np.cos(u), np.sin(u), 0 * u, color="#9ca3af", lw=0.7)  # equator
    ax.plot(np.cos(u), 0 * u, np.sin(u), color="#d1d5db", lw=0.7)  # xz meridian
    ax.plot(0 * u, np.cos(u), np.sin(u), color="#d1d5db", lw=0.7)  # yz meridian
    for axis, name in zip(np.eye(3), "xyz"):
        ax.plot(*np.stack([-axis, axis], axis=1), color="#9ca3af", lw=0.6, ls=":")
        ax.text(*(1.25 * axis), name, color="#4b5563", ha="center", va="center", fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    ax.set_title(title, fontsize=9)


def animate(result: dict, path: Path) -> Path:
    t = result["times"].cpu().numpy()
    S = result["entropy"].cpu().numpy() / math.log(2)  # in units of ln 2: 1 = maximally entangled
    C = result["concurrence"].cpu().numpy()
    bloch = [result["r1"].cpu().numpy(), result["r2"].cpu().numpy()]  # two × (T, 3)

    fig = plt.figure(figsize=(7, 4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])
    arrows, tips, trails = [], [], []
    for row, colour in enumerate([ACCENT[0], ACCENT[3]]):
        ax3 = fig.add_subplot(grid[row, 0], projection="3d")
        draw_sphere(ax3, f"qubit {row + 1}: Bloch vector of ρ{row + 1}")
        trails.append(ax3.plot([], [], [], color=colour, lw=0.8, alpha=0.5)[0])
        arrows.append(ax3.plot([], [], [], color=colour, lw=2.5)[0])
        tips.append(ax3.plot([], [], [], "o", color=colour, ms=4)[0])

    ax = fig.add_subplot(grid[:, 1])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("t")
    ax.set_ylabel("entanglement")
    (line_S,) = ax.plot([], [], color=ACCENT[1], lw=2, label="entropy S / ln 2")
    (line_C,) = ax.plot([], [], color=ACCENT[2], lw=1.5, ls="--", label="concurrence C")
    (line_r,) = ax.plot([], [], color=ACCENT[0], lw=1, alpha=0.7, label="|r₁| (Bloch vector length)")
    ax.legend(loc="lower right", fontsize=8)
    clock = ax.set_title("t = 0")
    r1_len = np.linalg.norm(bloch[0], axis=1)

    def update(step: int):
        for r, arrow, tip, trail in zip(bloch, arrows, tips, trails):
            trail.set_data_3d(r[: step + 1, 0], r[: step + 1, 1], r[: step + 1, 2])
            arrow.set_data_3d([0, r[step, 0]], [0, r[step, 1]], [0, r[step, 2]])
            tip.set_data_3d([r[step, 0]], [r[step, 1]], [r[step, 2]])
        line_S.set_data(t[: step + 1], S[: step + 1])
        line_C.set_data(t[: step + 1], C[: step + 1])
        line_r.set_data(t[: step + 1], r1_len[: step + 1])
        clock.set_text(f"t = {t[step]:.2f}   S = {S[step] * math.log(2):.3f}   C = {C[step]:.3f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
