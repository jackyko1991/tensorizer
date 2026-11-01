"""Animate the field: direction arrows on the grid, and E_x along the x axis split into its two parts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif

from .tensors import L


def animate(points, E, line, E_line, E_parts, path_q, charges, path: Path) -> Path:
    p = points.cpu().numpy()  # (G, 2)
    field = E.cpu().numpy()  # (T, G, 2)
    strength = np.linalg.norm(field, axis=-1)  # (T, G)
    direction = field / strength[..., None]  # unit arrows: only the direction is drawn
    x_line = line[:, 0].cpu().numpy()  # (n,)
    Ex_total = E_line[..., 0].cpu().numpy()  # (T, n)
    Ex_parts = E_parts[..., 0].cpu().numpy()  # (T, Q, n)
    Ex_total[np.abs(Ex_total) > 6] = np.nan  # don't draw the jump through ±∞ right next to a charge
    Ex_parts[np.abs(Ex_parts) > 6] = np.nan
    pos = path_q.cpu().numpy()  # (T, 2, 2)
    q = charges.cpu().numpy()  # (2,)
    colours = ["#dc2626" if value > 0 else "#2563eb" for value in q]

    fig, (ax, ax_line) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1, 1]})

    # left: arrows coloured by log |E|, and the two charges
    arrows = ax.quiver(
        p[:, 0], p[:, 1], direction[0, :, 0], direction[0, :, 1], np.log10(strength[0]),
        cmap="viridis", clim=(-1.0, 1.5), pivot="mid", scale=30, width=0.006,
    )  # fmt: skip
    dots = ax.scatter(pos[0, :, 0], pos[0, :, 1], c=colours, s=80, edgecolors="k", zorder=3)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color="#9ca3af", lw=0.8, ls=":")
    ax.set_title("direction of E (colour: log₁₀|E|)")

    # right: E_x along the dotted line y = 0 — the total is the sum of the two dashed curves
    part_lines = [
        ax_line.plot([], [], "--", lw=1.2, color=colours[k], label=f"charge {q[k]:+g} alone")[0] for k in range(2)
    ]
    (total_line,) = ax_line.plot([], [], color="k", lw=1.8, label="sum over q")
    ax_line.set_xlim(-L, L)
    ax_line.set_ylim(-3, 3)
    ax_line.axhline(0, color="#6b7280", lw=0.8)
    ax_line.set_xlabel("x  (along y = 0)")
    ax_line.set_ylabel("E_x")
    ax_line.set_title("superposition")
    ax_line.legend(loc="upper right", fontsize=8)

    def update(step: int):
        arrows.set_UVC(direction[step, :, 0], direction[step, :, 1], np.log10(strength[step]))
        dots.set_offsets(pos[step])
        for k, curve in enumerate(part_lines):
            curve.set_data(x_line, Ex_parts[step, k])
        total_line.set_data(x_line, Ex_total[step])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(pos), 72), path, fps=12)
