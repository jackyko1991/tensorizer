"""Animate every string in its own row: harmonics breathe in place, the plucked string's kink runs back and forth."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(times: torch.Tensor, x: torch.Tensor, traj: torch.Tensor, freqs: list[float], path: Path) -> Path:
    t = times.cpu().numpy()
    xs = x.cpu().numpy()
    u = traj.cpu().numpy()  # (T, B, N)
    n_rows = u.shape[1]

    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 1.1 + 1.0 * n_rows), sharex=True, squeeze=False)
    lines = []
    for b, ax in enumerate(axes[:, 0]):
        color = ACCENT[b % len(ACCENT)]
        ax.plot(xs, u[0, b], ls="--", color="#9ca3af", lw=1)  # starting shape
        if b < n_rows - 1:  # a harmonic swings between +u0 and −u0
            ax.plot(xs, -u[0, b], ls="--", color="#9ca3af", lw=1)
        lines.append(ax.plot(xs, u[0, b], color=color, lw=2)[0])
        ax.plot([xs[0], xs[-1]], [0, 0], "o", color="#374151", ms=4)  # fixed ends
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks([-1, 0, 1])
        name = "plucked" if b == n_rows - 1 else f"n = {b + 1},  f = {freqs[b]:.3f} Hz"
        ax.text(0.01, 0.92, name, transform=ax.transAxes, va="top", fontsize=9)
    axes[-1, 0].set_xlabel("x (m)")
    title = axes[0, 0].set_title("string displacement u(x, t)")  # placeholder so tight_layout leaves room

    def update(step: int):
        for b, line in enumerate(lines):
            line.set_ydata(u[step, b])
        title.set_text(f"string displacement u(x, t)    t = {t[step]:.2f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
