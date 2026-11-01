"""Animate light rays travelling to the flat mirror (left) and the concave mirror (right) and bouncing back."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(times: torch.Tensor, res: dict, radius: float, centre: torch.Tensor, path: Path) -> Path:
    t = times.numpy()
    R = radius
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
    for ax in (ax1, ax2):
        ax.set_xlim(-1.25 * R, 0.6 * R)
        ax.set_ylim(-1.0 * R, 1.0 * R)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    # flat mirror, lamp and its virtual image (mirror image of the lamp behind x = 0)
    ax1.axvline(0, color="#374151", lw=3)
    ax1.plot([-0.5 * R], [0.25 * R], "o", color="#ea580c", ms=8, label="lamp")
    ax1.plot([0.5 * R], [0.25 * R], "o", mfc="none", color="#ea580c", ms=8, label="virtual image")
    ax1.legend(loc="lower left", fontsize=7)
    ax1.set_title("flat mirror")

    # concave mirror: arc of the circle |p − c| = R near the vertex, plus C and F on the axis
    c = centre.numpy()
    phi = np.linspace(-1.1, 1.1, 100)
    ax2.plot(c[0] + R * np.cos(phi), c[1] + R * np.sin(phi), color="#374151", lw=3)
    ax2.axhline(0, color="#9ca3af", lw=0.8)
    ax2.plot([-R], [0], "k+", ms=10)
    ax2.text(-R, -0.9 * R, "C", ha="center", va="bottom")
    ax2.plot([-R / 2], [0], "k*", ms=10, zorder=5)
    ax2.text(-R / 2, -0.9 * R, "F = R/2", ha="center", va="bottom")
    ax2.vlines([-R, -R / 2], -0.8 * R, 0, colors="#9ca3af", lw=0.8, ls=":")
    ax2.set_title("concave mirror")

    def make_lines(ax, n):
        colors = plt.get_cmap("plasma")(np.linspace(0, 0.85, n))
        return [ax.plot([], [], lw=1.2, color=col)[0] for col in colors]

    n_rays = res["flat"].shape[1]
    flat_lines, curved_lines = make_lines(ax1, n_rays), make_lines(ax2, n_rays)
    ghosts = [ax1.plot([], [], ":", lw=0.8, color=line.get_color())[0] for line in flat_lines]

    def draw(lines, start, hits, t_hit, tip, now):  # start, hits: (R, 2), t_hit: (R,), tip: (R, 2)
        """Each ray is the polyline start → (hit point, once it has bounced) → current tip."""
        for r, line in enumerate(lines):
            if now < t_hit[r]:
                pts = np.array([start[r], tip[r]])
            else:
                pts = np.array([start[r], hits[r], tip[r]])
            line.set_data(pts[:, 0], pts[:, 1])

    flat_start, flat_hits, flat_t_hit, flat_tips = (
        res[k].numpy() for k in ("flat_start", "flat_hits", "flat_t_hit", "flat")
    )
    curved_start, curved_hits, curved_t_hit, curved_tips = (
        res[k].numpy() for k in ("curved_start", "curved_hits", "curved_t_hit", "curved")
    )
    image = np.array([0.5 * R, 0.25 * R])

    def update(step: int):
        now = t[step]
        draw(flat_lines, flat_start, flat_hits, flat_t_hit, flat_tips[step], now)
        draw(curved_lines, curved_start, curved_hits, curved_t_hit, curved_tips[step], now)
        for r, ghost in enumerate(ghosts):  # extend bounced rays backwards: they meet at the virtual image
            if now >= flat_t_hit[r]:
                ghost.set_data([flat_hits[r, 0], image[0]], [flat_hits[r, 1], image[1]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path, fps=15)
