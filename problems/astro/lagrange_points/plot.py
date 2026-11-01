"""Contours of the effective potential with the five Lagrange points, and particles librating around L4."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    times: torch.Tensor,
    traj: torch.Tensor,
    points: torch.Tensor,
    grid: torch.Tensor,
    phi: torch.Tensor,
    phi_points: torch.Tensor,
    mu: float,
    nudges: list[float],
    path: Path,
) -> Path:
    t = times.cpu().numpy()
    pos = traj[:, :, 0, :2].cpu().numpy()  # (T, B, 2)
    L = points[:, :2].cpu().numpy()  # (5, 2)
    X, Y = grid[..., 0].cpu().numpy(), grid[..., 1].cpu().numpy()
    phi = phi.cpu().numpy()
    phi_L = np.unique(phi_points.cpu().numpy())  # contour levels through the five L points
    levels = np.unique(np.concatenate([np.linspace(phi_L[0] - 0.4, phi_L[0], 5), phi_L, [phi_L[-1] + 0.01]]))

    fig, (ax, ax_z) = plt.subplots(1, 2, figsize=(8, 4))
    for a in (ax, ax_z):
        a.contourf(X, Y, np.clip(phi, levels[0], levels[-1]), levels=levels, cmap="Blues_r", alpha=0.4)
        a.contour(X, Y, phi, levels=phi_L, colors="#6b7280", linewidths=0.6)
        a.set_aspect("equal", adjustable="box")
        a.plot([-mu], [0], "o", ms=12, color="#f59e0b", zorder=5)  # star
        a.plot([1 - mu], [0], "o", ms=5, color="#1e40af", zorder=5)  # planet
        a.plot(L[:, 0], L[:, 1], "x", ms=7, mew=2, color=ACCENT[1], zorder=5)
        a.grid(False)
    for k, (x, y) in enumerate(L):
        ax.annotate(f"L{k + 1}", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8, color=ACCENT[1])
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x (rotating frame)")
    ax.set_ylabel("y")
    ax.set_title(f"Φ_eff contours, μ = {mu:g}", fontsize=9)
    zoom = 0.45
    cx, cy = L[3]
    ax_z.set_xlim(cx - zoom, cx + zoom)
    ax_z.set_ylim(cy - zoom, cy + zoom)
    ax_z.set_title("zoom on L4: tadpole orbits", fontsize=9)

    colors = [ACCENT[(b + 2) % len(ACCENT)] for b in range(pos.shape[1])]
    trails = [
        [a.plot([], [], lw=1.0, color=c, label=f"nudged {o:.1f}°")[0] for a in (ax, ax_z)] for c, o in zip(colors, nudges)
    ]
    dots = [[a.plot([], [], "o", ms=4, color=c)[0] for a in (ax, ax_z)] for c in colors]
    ax_z.legend(fontsize=7, loc="lower left")
    clock = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top", fontsize=9)

    def update(step: int):
        for b in range(pos.shape[1]):
            for line in trails[b]:
                line.set_data(pos[: step + 1, b, 0], pos[: step + 1, b, 1])
            for dot in dots[b]:
                dot.set_data([pos[step, b, 0]], [pos[step, b, 1]])
        clock.set_text(f"t = {t[step]:.0f}   ({t[step] / (2 * np.pi):.1f} planet orbits)")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)

