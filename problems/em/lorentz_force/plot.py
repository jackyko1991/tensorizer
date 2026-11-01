"""Animate the helices in 3D: each particle leaves a trail, labelled by its charge and mass."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(times, traj, q, m, drift: torch.Tensor, path: Path) -> Path:
    pos = traj[:, :, 0, :].cpu().numpy()  # (T, P, 3)
    t = times.cpu().numpy()

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(projection="3d")
    lo, hi = pos.min(axis=(0, 1)), pos.max(axis=(0, 1))
    pad = 0.05 * (hi - lo).max() + 0.1
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_xlabel("x  (E×B)")
    ax.set_ylabel("y  (E)")
    ax.set_zlabel("z  (B)")
    ax.view_init(elev=22, azim=-60)

    labels = [f"q={qb:+.0f}, m={mb:g}" for qb, mb in zip(q.tolist(), m.tolist())]
    lines = [ax.plot([], [], [], lw=1.4, label=lab)[0] for lab in labels]
    dots = [ax.plot([], [], [], "o", ms=5, color=line.get_color())[0] for line in lines]
    ax.legend(loc="upper left", fontsize=7)
    u = drift.tolist()
    ax.set_title(f"E×B drift  u = ({u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f})")
    clock = ax.text2D(0.85, 0.95, "", transform=ax.transAxes, va="top")

    def update(step: int):
        for b, (line, dot) in enumerate(zip(lines, dots)):
            line.set_data_3d(pos[: step + 1, b, 0], pos[: step + 1, b, 1], pos[: step + 1, b, 2])
            dot.set_data_3d([pos[step, b, 0]], [pos[step, b, 1]], [pos[step, b, 2]])
        clock.set_text(f"t = {t[step]:.1f}")

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.93)
    return save_gif(fig, update, frame_indices(len(t), 90), path)  # 3D frames are slow: 90 is plenty
