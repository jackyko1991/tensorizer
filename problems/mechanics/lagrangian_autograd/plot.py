"""Animate the spring pendulum with a trail, next to r(t) and θ(t)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif


def spring_points(tip: np.ndarray, n_coils: int = 10, width: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """A zigzag from the pivot (0, 0) to tip = (x, y), drawn across the line joining them."""
    s = np.linspace(0, 1, 2 * n_coils + 3)
    offset = np.zeros_like(s)
    offset[1:-1:2], offset[2:-1:2] = width, -width
    offset[-2] = 0.0
    side = np.array([-tip[1], tip[0]]) / np.linalg.norm(tip)  # unit vector across the spring
    return s * tip[0] + offset * side[0], s * tip[1] + offset * side[1]


def animate(sol: dict, rest_length: float, path: Path, trail: float = 0.25) -> Path:
    """trail = fraction of the run that the bob's tail remembers."""
    t = sol["times"].cpu().numpy()  # (T,)
    r, theta = sol["q"][:, 0].cpu().numpy(), sol["q"][:, 1].cpu().numpy()  # (T,) each
    bob = np.stack([r * np.sin(theta), -r * np.cos(theta)], axis=-1)  # (T, 2) Cartesian position
    E = sol["E"].cpu().numpy()
    tail = max(2, int(trail * len(t)))
    reach = 1.1 * np.abs(bob).max()

    fig = plt.figure(figsize=(7, 4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
    ax_swing = fig.add_subplot(grid[:, 0])
    ax_r = fig.add_subplot(grid[0, 1])
    ax_theta = fig.add_subplot(grid[1, 1], sharex=ax_r)

    # --- left: the pendulum ---
    ax_swing.set_xlim(-reach, reach)
    ax_swing.set_ylim(-reach, 0.25 * reach)
    ax_swing.set_aspect("equal", adjustable="box")
    ax_swing.set_xticks([])
    ax_swing.set_yticks([])
    ax_swing.plot([0], [0], "k^", ms=6)
    path_line = ax_swing.plot([], [], lw=1, color=ACCENT[0], alpha=0.5)[0]
    spring = ax_swing.plot([], [], lw=1.2, color="#6b7280")[0]
    dot = ax_swing.plot([], [], "o", ms=10, color=ACCENT[1])[0]
    clock = ax_swing.text(0.03, 0.97, "", transform=ax_swing.transAxes, va="top", fontsize=9)
    ax_swing.set_title(f"E = {E[0]:.3f} J (drift {np.abs(E - E[0]).max():.0e})", fontsize=8)

    # --- right: the two generalised coordinates ---
    ax_r.set_xlim(t[0], t[-1])
    ax_r.set_ylim(r.min() - 0.05, r.max() + 0.05)
    ax_r.axhline(rest_length, color="#9ca3af", lw=0.8, ls="--")
    ax_r.set_ylabel("r (m)")
    ax_theta.set_ylim(np.degrees(theta).min() - 5, np.degrees(theta).max() + 5)
    ax_theta.set_ylabel("θ (°)")
    ax_theta.set_xlabel("t (s)")
    plt.setp(ax_r.get_xticklabels(), visible=False)
    r_line = ax_r.plot([], [], lw=1.3, color=ACCENT[2])[0]
    theta_line = ax_theta.plot([], [], lw=1.3, color=ACCENT[3])[0]

    def update(step: int):
        start = max(0, step - tail)
        path_line.set_data(bob[start : step + 1, 0], bob[start : step + 1, 1])
        spring.set_data(*spring_points(bob[step]))
        dot.set_data([bob[step, 0]], [bob[step, 1]])
        r_line.set_data(t[: step + 1], r[: step + 1])
        theta_line.set_data(t[: step + 1], np.degrees(theta[: step + 1]))
        clock.set_text(f"t = {t[step]:.2f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 120), path)
