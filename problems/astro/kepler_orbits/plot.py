"""Animate the orbits with equal-time sectors (2nd law) next to a T² vs a³ plot (3rd law)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    times: torch.Tensor,
    traj: torch.Tensor,
    a: torch.Tensor,
    e: torch.Tensor,
    periods: torch.Tensor,
    path: Path,
    n_slices: int = 12,
) -> Path:
    t = times.cpu().numpy()  # (T,)
    pos = traj[:, :, 0, :].cpu().numpy()  # (T, B, 2)
    a, e, periods = a.cpu().numpy(), e.cpu().numpy(), periods.cpu().numpy()
    B = pos.shape[1]
    hb = B - 1  # highlight the most eccentric planet with equal-time sectors

    fig, (ax, ax_k) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1.35, 1]})
    lo, hi = pos.reshape(-1, 2).min(0), pos.reshape(-1, 2).max(0)
    pad = 0.08 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.plot([0], [0], "*", ms=14, color="#f59e0b", zorder=5)  # the star sits at a focus

    # Kepler II: slices of equal duration T/n_slices for the highlighted planet, every other one shaded
    T_h = periods[hb] if np.isfinite(periods[hb]) else t[-1]
    edges = np.searchsorted(t, np.arange(n_slices + 1) * T_h / n_slices)
    sectors = []
    for k in range(0, n_slices, 2):
        i0, i1 = edges[k], min(edges[k + 1], len(t) - 1)
        corners = np.vstack([[0.0, 0.0], pos[i0 : i1 + 1, hb]])
        patch = Polygon(corners, closed=True, color=ACCENT[hb % len(ACCENT)], alpha=0.25, lw=0, visible=False)
        ax.add_patch(patch)
        sectors.append((i1, patch))

    lines = [ax.plot([], [], lw=1.4, color=ACCENT[b % len(ACCENT)])[0] for b in range(B)]
    dots = [ax.plot([], [], "o", ms=6, color=ACCENT[b % len(ACCENT)])[0] for b in range(B)]
    ax.set_title(f"shaded sectors: equal Δt = T/{n_slices} → equal areas", fontsize=9)
    clock = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top", fontsize=9)

    # Kepler III: T² against a³ (years and AU, so the slope is exactly 1)
    a3 = a**3
    top = 1.15 * max(a3.max(), np.nanmax(periods**2) if np.isfinite(periods).any() else 1.0)
    ax_k.plot([0, top], [0, top], "--", color="#9ca3af", lw=1, label="T² = a³")
    marks = [
        ax_k.plot([], [], "o", ms=8, color=ACCENT[b % len(ACCENT)], label=f"a={a[b]:.1f}, e={e[b]:.2f}")[0]
        for b in range(B)
    ]
    ax_k.set_xlim(0, top)
    ax_k.set_ylim(0, top)
    ax_k.set_xlabel("a³ (AU³)")
    ax_k.set_ylabel("T² (yr²)  measured")
    ax_k.legend(fontsize=7, loc="upper left")
    ax_k.set_title("third law: dot appears after one orbit", fontsize=9)

    def update(step: int):
        for b in range(B):
            lines[b].set_data(pos[: step + 1, b, 0], pos[: step + 1, b, 1])
            dots[b].set_data([pos[step, b, 0]], [pos[step, b, 1]])
            done = np.isfinite(periods[b]) and t[step] >= periods[b]
            marks[b].set_data([a3[b]] if done else [], [periods[b] ** 2] if done else [])
        for end, patch in sectors:
            patch.set_visible(step >= end)
        clock.set_text(f"t = {t[step]:.2f} yr")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
