"""Animate the launches around the planet next to the distance r(t) of every rocket."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(
    times: torch.Tensor,
    traj: torch.Tensor,
    fractions: torch.Tensor,
    r_max: torch.Tensor | None,
    R: float,
    path: Path,
    view: float = 7.0,
) -> Path:
    """r_max: predicted highest points (B,) for vertical launches, or None when the launches are tilted."""
    t = times.cpu().numpy()
    pos = traj[:, :, 0, :].cpu().numpy() / R  # (T, B, 2) in planet radii
    r = np.linalg.norm(pos, axis=-1)  # (T, B)
    f = fractions.cpu().numpy()
    # blue shades = bound (v < v_esc), red shades = unbound (v ≥ v_esc)
    colors = [
        plt.cm.Blues(np.interp(fb, [f.min(), 1], [0.4, 0.95])) if fb < 1 - 1e-6 else plt.cm.Reds(0.5 + 0.4 * (fb - 1))
        for fb in f
    ]
    B = len(f)

    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax.set_xlim(-view, view)
    ax.set_ylim(-0.3 * view, view)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.add_patch(plt.Circle((0, 0), 1.0, color="#93c5fd", ec="#1e40af", lw=1.5, zorder=3))
    lines = [ax.plot([], [], lw=1.3, color=colors[b])[0] for b in range(B)]
    dots = [ax.plot([], [], "o", ms=5, color=colors[b], zorder=4)[0] for b in range(B)]
    clock = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top", fontsize=9)
    dark_star = "dark star: v_esc = c  ⇔  R = 2GM/c²"
    ax.text(0.97, 0.03, dark_star, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#374151")
    ax.set_title("blue: bound (E < 0)   red: escapes (E ≥ 0)", fontsize=9)

    top = 12.0
    ax_r.set_xlim(t[0], t[-1])
    ax_r.set_ylim(0, top)
    ax_r.set_xlabel("t  (units of √(R³/GM))")
    ax_r.set_ylabel("r / R")
    ax_r.axhline(1.0, color="#1e40af", lw=1)
    curves = [ax_r.plot([], [], lw=1.4, color=colors[b], label=f"{f[b]:.2f}")[0] for b in range(B)]
    if r_max is not None:
        for b, rm in enumerate(r_max.cpu().numpy() / R):
            if np.isfinite(rm) and 0 < rm < top:
                ax_r.axhline(rm, color=colors[b], lw=0.8, ls="--")
        ax_r.set_title("dashed: r_max from energy conservation", fontsize=9)
    ax_r.legend(fontsize=7, ncol=2, loc="upper right", title="v / v_esc", title_fontsize=7)

    def update(step: int):
        for b in range(B):
            lines[b].set_data(pos[: step + 1, b, 0], pos[: step + 1, b, 1])
            dots[b].set_data([pos[step, b, 0]], [pos[step, b, 1]])
            curves[b].set_data(t[: step + 1], r[: step + 1, b])
        clock.set_text(f"t = {t[step]:.1f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
