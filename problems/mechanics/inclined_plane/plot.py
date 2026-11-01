"""Animate one block per ramp, side by side, with the distance slid along each slope below."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(sol: dict, theta: torch.Tensor, ramp_length: float, path: Path) -> Path:
    t = sol["times"].cpu().numpy()  # (T,)
    d = sol["d"].cpu().numpy()  # (T, B)
    a_par = sol["a_par"].cpu().numpy()  # (B,)
    angle = theta.cpu().numpy()  # (B,)
    n_ramps = len(angle)
    L, size = ramp_length, 0.12 * ramp_length  # ramp length and block edge

    s = np.stack([np.cos(angle), -np.sin(angle)], axis=-1)  # (B, 2) down the slope
    n = np.stack([np.sin(angle), np.cos(angle)], axis=-1)  # (B, 2) out of the surface
    widths = L * np.cos(angle)
    left = np.concatenate([[0.0], np.cumsum(widths + 0.25 * L)[:-1]])  # x of each ramp's top corner
    top = np.stack([left, L * np.sin(angle)], axis=-1)  # (B, 2) where each block starts

    fig, (ax_ramp, ax_dist) = plt.subplots(2, 1, figsize=(7, 4.6), gridspec_kw={"height_ratios": [1.3, 1]})

    # --- top: the ramps and blocks ---
    ax_ramp.set_xlim(-0.1 * L, left[-1] + widths[-1] + 0.1 * L)
    ax_ramp.set_ylim(-0.25 * L, L * np.sin(angle).max() + 0.3 * L)
    ax_ramp.set_aspect("equal", adjustable="box")
    ax_ramp.axis("off")
    blocks = []
    for b in range(n_ramps):
        color = ACCENT[b % len(ACCENT)]
        corners = [top[b], top[b] + L * s[b], [left[b], 0.0]]  # right-angled triangle
        ax_ramp.add_patch(Polygon(corners, closed=True, color="#e5e7eb", ec="#6b7280"))
        state = "slides" if a_par[b] > 1e-6 else "at rest"
        ax_ramp.text(
            left[b] + 0.4 * widths[b], -0.2 * L, f"{np.degrees(angle[b]):.0f}°  {state}", fontsize=7, ha="center"
        )
        blocks.append(ax_ramp.add_patch(Polygon(np.zeros((4, 2)), closed=True, color=color)))
    clock = ax_ramp.text(0.0, 1.0, "", transform=ax_ramp.transAxes, va="top", fontsize=9)

    # --- bottom: distance along the slope vs time ---
    ax_dist.set_xlim(t[0], t[-1])
    ax_dist.set_ylim(0, L * 1.08)
    ax_dist.axhline(L, color="#9ca3af", lw=0.8, ls="--")
    ax_dist.set_xlabel("t (s)")
    ax_dist.set_ylabel("distance slid (m)")
    curves = [ax_dist.plot([], [], lw=1.6, color=ACCENT[b % len(ACCENT)])[0] for b in range(n_ramps)]

    def update(step: int):
        for b in range(n_ramps):
            centre = top[b] + d[step, b] * s[b] + 0.5 * size * n[b]
            half_s, half_n = 0.5 * size * s[b], 0.5 * size * n[b]
            blocks[b].set_xy(
                [centre - half_s - half_n, centre + half_s - half_n, centre + half_s + half_n, centre - half_s + half_n]
            )
            curves[b].set_data(t[: step + 1], d[: step + 1, b])
        clock.set_text(f"t = {t[step]:.2f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
