"""Animate the pendulums swinging side by side, with θ(t) traces against the small-angle cosine."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(times: torch.Tensor, traj: torch.Tensor, small: torch.Tensor, L: float, path: Path) -> Path:
    t = times.cpu().numpy()  # (T,)
    theta = traj[:, :, 0].cpu().numpy()  # (T, B)
    theta_small = small.cpu().numpy()  # (T, B)
    n = theta.shape[1]
    pivots = 2.2 * L * np.arange(n)  # x position of each pivot

    fig, (ax_swing, ax_trace) = plt.subplots(2, 1, figsize=(7, 4.6), gridspec_kw={"height_ratios": [1.1, 1]})

    # --- top: the pendulums ---
    ax_swing.set_xlim(-1.2 * L, pivots[-1] + 1.2 * L)
    ax_swing.set_ylim(-1.15 * L, 1.35 * L)
    ax_swing.set_aspect("equal", adjustable="box")
    ax_swing.set_xticks([])
    ax_swing.set_yticks([])
    ax_swing.grid(False)
    ax_swing.plot(pivots, np.zeros(n), "k^", ms=6)
    rods = [ax_swing.plot([], [], color="#6b7280", lw=1.5)[0] for _ in range(n)]
    bobs = [ax_swing.plot([], [], "o", ms=9)[0] for _ in range(n)]
    for b in range(n):
        label = f"{np.degrees(theta[0, b]):.0f}°"
        ax_swing.text(pivots[b] + 0.15 * L, 0.1 * L, label, fontsize=8, color=bobs[b].get_color())
    clock = ax_swing.text(0.02, 0.97, "", transform=ax_swing.transAxes, va="top", fontsize=9)

    # --- bottom: θ(t), solid = full equation, dashed = small-angle solution ---
    ax_trace.set_xlim(t[0], t[-1])
    ax_trace.set_ylim(-np.degrees(np.abs(theta).max()) * 1.1, np.degrees(np.abs(theta).max()) * 1.1)
    ax_trace.set_xlabel("t (s)")
    ax_trace.set_ylabel("θ (°)")
    ax_trace.set_title("solid: full equation (sin θ)    dashed: small-angle (sin θ ≈ θ)", fontsize=9)
    traces = [ax_trace.plot([], [], lw=1.4, color=bob.get_color())[0] for bob in bobs]
    linear = [ax_trace.plot([], [], lw=0.9, ls="--", color=bob.get_color())[0] for bob in bobs]

    def update(step: int):
        for b in range(n):
            x_bob = pivots[b] + L * np.sin(theta[step, b])
            y_bob = -L * np.cos(theta[step, b])
            rods[b].set_data([pivots[b], x_bob], [0.0, y_bob])
            bobs[b].set_data([x_bob], [y_bob])
            traces[b].set_data(t[: step + 1], np.degrees(theta[: step + 1, b]))
            linear[b].set_data(t[: step + 1], np.degrees(theta_small[: step + 1, b]))
        clock.set_text(f"t = {t[step]:.2f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 100), path)
