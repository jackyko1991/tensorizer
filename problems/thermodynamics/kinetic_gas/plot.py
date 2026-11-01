"""Animate the disks in the box next to their speed histogram and the Maxwell–Boltzmann curve."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import maxwell_boltzmann_2d


def animate(
    times: torch.Tensor,
    xs: torch.Tensor,
    vs: torch.Tensor,
    kT: float,
    box: float,
    radius: float,
    title: str,
    path: Path,
) -> Path:
    t = times.cpu().numpy()
    pos = xs.cpu().numpy()  # (F, N, 2)
    speed = vs.norm(dim=-1).cpu().numpy()  # (F, N)
    v_top = 3.2 * np.sqrt(kT)  # right edge of the histogram
    bins = np.linspace(0, v_top, 26)
    v_curve = torch.linspace(0, v_top, 200)
    f_curve = maxwell_boltzmann_2d(v_curve, kT).numpy()

    fig, (ax, ax_h) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1, 1.1]})
    ax.set_xlim(0, box)
    ax.set_ylim(0, box)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    marker = (2 * radius / box * 3.0 * 72) ** 2  # disk diameter in points², box ≈ 3 inches tall
    dots = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=marker, c=speed[0], cmap="viridis", vmin=0, vmax=v_top)
    clock = ax.set_title("")

    hist = ax_h.hist(speed[0], bins=bins, density=True, color=ACCENT[0], alpha=0.6, label="simulation")[2]
    ax_h.plot(v_curve.numpy(), f_curve, color=ACCENT[1], lw=2, label="Maxwell–Boltzmann")
    ax_h.set_xlim(0, v_top)
    ax_h.set_ylim(0, 1.6 * f_curve.max())
    ax_h.set_xlabel("speed |v|")
    ax_h.set_ylabel("probability density")
    ax_h.set_title(title, fontsize=9)
    ax_h.legend(loc="upper right", fontsize=8)

    def update(step: int):
        dots.set_offsets(pos[step])
        dots.set_array(speed[step])
        heights, _ = np.histogram(speed[step], bins=bins, density=True)
        for bar, h in zip(hist, heights):
            bar.set_height(h)
        clock.set_text(f"t = {t[step]:.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path)
