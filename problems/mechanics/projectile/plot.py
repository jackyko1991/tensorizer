"""Animate the trajectories: each ball traces its path, labelled by launch angle."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(times: torch.Tensor, traj: torch.Tensor, angles: torch.Tensor, path: Path) -> Path:
    pos = traj[:, :, 0, :].cpu().numpy()  # (T, B, 2)
    t = times.cpu().numpy()
    degrees = torch.rad2deg(angles).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, max(1.0, pos[..., 0].max() * 1.05))
    ax.set_ylim(0, max(1.0, pos[..., 1].max() * 1.15))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="box")

    lines = [ax.plot([], [], lw=1.8, label=f"{d:.0f}°")[0] for d in degrees]
    dots = [ax.plot([], [], "o", color=line.get_color())[0] for line in lines]
    clock = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right", fontsize=8, title="launch angle")

    def update(step: int):
        for b, (line, dot) in enumerate(zip(lines, dots)):
            line.set_data(pos[: step + 1, b, 0], pos[: step + 1, b, 1])
            dot.set_data([pos[step, b, 0]], [pos[step, b, 1]])
        clock.set_text(f"t = {t[step]:.2f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path)
