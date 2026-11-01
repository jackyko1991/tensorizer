"""Darts landing in the square next to every experiment's error shrinking like 1/√N."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(x: torch.Tensor, est: torch.Tensor, sd: torch.Tensor, path: Path) -> Path:
    """x (B, N, 2), est (B, N), sd (N,)."""
    pts = x[0].cpu().numpy()
    inside = (pts**2).sum(-1) <= 1
    steps = np.unique(np.geomspace(10, est.shape[1], 80).astype(int))  # log-spaced frame sizes
    curve = np.unique(np.geomspace(1, est.shape[1], 400).astype(int))  # log-spaced points for the error curves
    n_all = curve
    err = np.abs(est.cpu().numpy()[:, curve - 1] - math.pi)
    sd = sd[torch.as_tensor(curve - 1)]
    shown = min(len(pts), 4000)  # draw at most this many darts

    fig, (ax_d, ax_e) = plt.subplots(1, 2, figsize=(9.5, 4.3), gridspec_kw={"width_ratios": [1, 1.3]})
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.set_aspect("equal")
    t = np.linspace(0, math.pi / 2, 100)
    ax_d.plot(np.cos(t), np.sin(t), color="#374151", lw=1.5)
    in_dots = ax_d.scatter([], [], s=3, color="#2563eb")
    out_dots = ax_d.scatter([], [], s=3, color="#f59e0b")
    title = ax_d.set_title("", fontsize=10, family="monospace")

    for b in range(err.shape[0]):
        ax_e.loglog(n_all, np.maximum(err[b], 1e-7), color="#9ca3af", lw=0.6, alpha=0.6)
    ax_e.loglog(n_all, sd.cpu().numpy(), color="#2563eb", lw=2, label="spread over experiments")
    ax_e.loglog(n_all, np.sqrt(math.pi * (4 - math.pi) / n_all), color="#f59e0b", ls="--", lw=1.5, label="√(π(4−π)/N)")
    marker = ax_e.axvline(10, color="#dc2626", lw=1)
    ax_e.set_ylim(1e-5, 2)
    ax_e.set_xlabel("darts N")
    ax_e.set_ylabel("|π̂ − π|")
    ax_e.set_title("error of each experiment (grey)", fontsize=10)
    ax_e.legend(fontsize=8, frameon=False, loc="lower left")
    ax_e.grid(True, which="both", alpha=0.2)

    def update(k: int):
        n = steps[k]
        m = min(n, shown)
        in_dots.set_offsets(pts[:m][inside[:m]])
        out_dots.set_offsets(pts[:m][~inside[:m]])
        title.set_text(f"N = {n:7d}   π̂ = {est[0, n - 1].item():.5f}")
        marker.set_xdata([n, n])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(steps), 80), path, fps=10)
