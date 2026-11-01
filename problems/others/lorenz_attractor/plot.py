"""Left: the butterfly in 3D, with each trajectory's recent path in its own colour. Right: log10 of their spread."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    times: torch.Tensor, traj: torch.Tensor, s: torch.Tensor, fit: tuple, radius: float, path: Path, trail: float = 1.5
) -> Path:
    """traj: (T, B, 3); s: (T,) spread; fit = (λ, t0, t1, c). trail = time units of path behind each point."""
    t = times.cpu().numpy()  # (T,)
    x = traj.cpu().numpy()  # (T, B, 3)
    log_s = s.clamp_min(1e-300).log10().cpu().numpy()  # (T,)
    n_traj = x.shape[1]
    tail = max(2, int(trail / (t[1] - t[0])))
    frames = frame_indices(len(t), 90)

    fig = plt.figure(figsize=(7.6, 4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax_s = fig.add_subplot(1, 2, 2)

    # --- left: the attractor (grey ghost of trajectory 0) and the coloured trajectories ---
    ghost = x[:: max(1, len(t) // 3000), 0]  # thinned path of one trajectory: the shape of the attractor
    ax.plot(ghost[:, 0], ghost[:, 1], ghost[:, 2], lw=0.4, color="#d1d5db")
    lo, hi = x.min(axis=(0, 1)), x.max(axis=(0, 1))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.tick_params(labelsize=7)
    colors = [ACCENT[b % len(ACCENT)] for b in range(n_traj)]
    lines = [ax.plot([], [], [], lw=1.2, color=c)[0] for c in colors]
    dots = [ax.plot([], [], [], "o", ms=4, color=c)[0] for c in colors]
    clock = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)
    ax.view_init(elev=18, azim=-60)
    ax.set_box_aspect(None, zoom=0.88)  # leave room for the z label

    # --- right: how far apart the trajectories are, in powers of ten ---
    ax_s.set_xlim(t[0], t[-1])
    ax_s.set_ylim(log_s.min() - 0.5, log_s.max() + 0.8)
    ax_s.set_xlabel("t")
    ax_s.set_ylabel("log₁₀ spread of the trajectories")
    ax_s.set_title(f"{n_traj} starts within r = {radius:.0e}", fontsize=9)
    trace = ax_s.plot([], [], lw=1.2, color="#111827")[0]
    lam, t0, t1, c = fit
    if lam == lam:  # not nan
        ln10 = 2.302585
        ax_s.plot([t0, t1], [(c + lam * t0) / ln10, (c + lam * t1) / ln10], "--", color="#f59e0b", lw=2)
        ax_s.text(0.03, 0.97, f"fit: λ ≈ {lam:.2f}", transform=ax_s.transAxes, va="top", color="#b45309")

    def update(step: int):
        start = max(0, step - tail)
        for b in range(n_traj):
            seg = x[start : step + 1, b]  # (≤tail, 3) recent path of trajectory b
            lines[b].set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
            dots[b].set_data_3d([x[step, b, 0]], [x[step, b, 1]], [x[step, b, 2]])
        clock.set_text(f"t = {t[step]:.1f}")
        trace.set_data(t[: step + 1], log_s[: step + 1])

    fig.tight_layout()
    return save_gif(fig, update, frames, path)
