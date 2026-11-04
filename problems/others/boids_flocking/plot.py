"""A flock of arrows forming (Vicsek and boids) beside the order parameter φ against the noise η."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(
    shown: dict, boid: dict, eta: torch.Tensor, phi: torch.Tensor, late: torch.Tensor, box: float, path: Path
) -> Path:
    """shown: Vicsek run at one noise (x (T, N, 2), theta (T, N), phi (T,)); boid: x, v (T, N, 2), phi (T,);
    eta (B,), phi (B, T), late (B,) for the batch of noise levels."""
    xv, th = shown["x"].numpy(), shown["theta"].numpy()
    xb, vb = boid["x"].numpy(), boid["v"].numpy()
    ub = vb / np.linalg.norm(vb, axis=-1, keepdims=True)
    e, p, avg = eta.numpy(), phi.numpy(), late.numpy()
    n_t = len(th)

    fig = plt.figure(figsize=(11, 3.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.28)
    panels = []
    for g, name in ((grid[0], f"Vicsek, η = {shown['eta']:.2f}"), (grid[1], "Reynolds boids")):
        ax = fig.add_subplot(g)
        ax.set_xlim(0, box)
        ax.set_ylim(0, box)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        panels.append((ax, ax.set_title(name, fontsize=10), name))
    q_v = panels[0][0].quiver(
        xv[0, :, 0],
        xv[0, :, 1],
        np.cos(th[0]),
        np.sin(th[0]),
        th[0],
        cmap="twilight",
        clim=(-np.pi, np.pi),
        scale=18,
        width=0.007,
    )
    q_b = panels[1][0].quiver(
        xb[0, :, 0], xb[0, :, 1], ub[0, :, 0], ub[0, :, 1], color="#2563eb", scale=18, width=0.006
    )

    ax_p = fig.add_subplot(grid[2])
    ax_p.plot(e, avg, color="#d1d5db", lw=2, label="time average (end)")
    (now,) = ax_p.plot(e, p[:, 0], "o-", color="#dc2626", ms=4, lw=1.2, label="φ now, every η at once")
    ax_p.axvline(shown["eta"], color="#6b7280", ls=":", lw=1)
    ax_p.axhline(1 / np.sqrt(xv.shape[1]), color="#f59e0b", ls="--", lw=1, label="1/√N (no order)")
    ax_p.set_xlim(0, 2 * np.pi)
    ax_p.set_ylim(0, 1.03)
    ax_p.set_xlabel("noise η")
    ax_p.set_ylabel("polar order φ = |Σ v| / Σ |v|")
    ax_p.set_title("order–disorder transition", fontsize=10)
    ax_p.legend(fontsize=7, frameon=False, loc="upper right")
    clock = ax_p.text(0.03, 0.05, "", transform=ax_p.transAxes, fontsize=8, color="#6b7280")

    def update(t: int):
        q_v.set_offsets(xv[t])
        q_v.set_UVC(np.cos(th[t]), np.sin(th[t]), th[t])
        q_b.set_offsets(xb[t])
        q_b.set_UVC(ub[t, :, 0], ub[t, :, 1])
        panels[0][1].set_text(f"{panels[0][2]}: φ = {shown['phi'][t]:.2f}")
        panels[1][1].set_text(f"{panels[1][2]}: φ = {boid['phi'][t]:.2f}")
        now.set_ydata(p[:, t])
        clock.set_text(f"step {t}")

    return save_gif(fig, update, frame_indices(n_t, 50), path, fps=6)
