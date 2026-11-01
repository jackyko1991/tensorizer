"""Animate the Bloch vector on the Bloch sphere (left) and its components s_k(t) (right)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def draw_sphere(ax) -> None:
    """Faint unit sphere with equator, a meridian and the x, y, z axes."""
    u = np.linspace(0, 2 * np.pi, 60)
    ax.plot(np.cos(u), np.sin(u), 0 * u, color="#9ca3af", lw=0.7)  # equator
    ax.plot(np.cos(u), 0 * u, np.sin(u), color="#d1d5db", lw=0.7)  # xz meridian
    ax.plot(0 * u, np.cos(u), np.sin(u), color="#d1d5db", lw=0.7)  # yz meridian
    for axis, name in zip(np.eye(3), "xyz"):
        ax.plot(*np.stack([-axis, axis], axis=1), color="#9ca3af", lw=0.6, ls=":")
        ax.text(*(1.2 * axis), name, color="#4b5563", ha="center", va="center")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)


def animate(times: torch.Tensor, s: torch.Tensor, path: Path) -> Path:
    s = s.cpu().numpy()  # (T, 3)
    t = times.cpu().numpy()

    fig = plt.figure(figsize=(7, 4))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    draw_sphere(ax3)
    (trail,) = ax3.plot([], [], [], color=ACCENT[0], lw=1, alpha=0.6)
    (arrow,) = ax3.plot([], [], [], color=ACCENT[1], lw=2.5)
    (tip,) = ax3.plot([], [], [], "o", color=ACCENT[1], ms=5)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("t")
    ax.set_ylabel("Bloch vector s_k")
    curves = [ax.plot([], [], lw=1.4, label=f"s_{k}")[0] for k in "xyz"]
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    clock = ax.set_title("t = 0")  # placeholder so tight_layout leaves room

    def update(step: int):
        trail.set_data_3d(s[: step + 1, 0], s[: step + 1, 1], s[: step + 1, 2])
        arrow.set_data_3d([0, s[step, 0]], [0, s[step, 1]], [0, s[step, 2]])
        tip.set_data_3d([s[step, 0]], [s[step, 1]], [s[step, 2]])
        for k, curve in enumerate(curves):
            curve.set_data(t[: step + 1], s[: step + 1, k])
        clock.set_text(f"t = {t[step]:.1f}   |s| = {np.linalg.norm(s[step]):.4f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path)
