"""3D animation next to the conservation checks: each body leaves a fading trail (marker size grows with mass, the
camera turns slowly to show depth), and the right panel draws the energy change ΔE/|E₀| and total momentum |P|."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    times: torch.Tensor,
    xs: torch.Tensor,
    m: torch.Tensor,
    path: Path,
    trail: float = 0.3,
    drift: torch.Tensor | None = None,
    p_norm: torch.Tensor | None = None,
) -> Path:
    """trail = fraction of the run that each body's tail remembers; drift (T,) = ΔE/|E₀| and p_norm (T,) = |P| are
    plotted against time (the conservation laws the run must obey)."""
    pos = xs.cpu().numpy()  # (T, N, 3)
    t = times.cpu().numpy()  # (T,)
    mass = m.cpu().numpy()  # (N,)
    tail = max(2, int(trail * len(t)))
    checks = drift is not None and p_norm is not None

    fig = plt.figure(figsize=(9.5, 4.6) if checks else (6, 5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.3, 1]) if checks else fig.add_gridspec(1, 1)
    ax = fig.add_subplot(grid[0], projection="3d")
    half = abs(pos).max() * 1.05
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{len(mass)} bodies, G = 1", fontsize=10)

    # a body holding most of the total mass is drawn as a yellow star
    colors = ["#f59e0b" if mi > 0.5 * mass.sum() else ACCENT[a % len(ACCENT)] for a, mi in enumerate(mass)]
    sizes = 4 + 8 * (mass / mass.max()) ** (1 / 3)  # radius ∝ m^(1/3)
    trails = [ax.plot([], [], [], lw=1.2, color=c, alpha=0.7)[0] for c in colors]
    dots = [ax.plot([], [], [], "o", ms=s, color=c)[0] for s, c in zip(sizes, colors)]
    clock = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    if checks:
        # --- right: conservation checks on a log scale (ideal: flat and tiny) ---
        e_err = np.abs(drift.cpu().numpy()) + 1e-17  # (T,) |ΔE| / |E₀|
        p_err = p_norm.cpu().numpy() + 1e-17  # (T,) |P|
        ax_c = fig.add_subplot(grid[1])
        ax_c.set_yscale("log")
        ax_c.set_xlim(t[0], t[-1])
        top = max(e_err.max(), p_err.max())
        ax_c.set_ylim(min(e_err[1:].min(), p_err.min(), top / 1e3) / 3, top * 10)
        ax_c.set_xlabel("t (time units)")
        ax_c.set_title("conservation checks", fontsize=10)
        ax_c.grid(True, which="major", alpha=0.25)
        e_line = ax_c.plot([], [], lw=1.4, color=ACCENT[0], label="energy change |ΔE| / |E₀|")[0]
        p_line = ax_c.plot([], [], lw=1.2, color=ACCENT[2], label="total momentum |P|")[0]
        ax_c.legend(loc="upper left", fontsize=8, framealpha=0.9)

    def update(step: int):
        start = max(0, step - tail)
        for a, (line, dot) in enumerate(zip(trails, dots)):
            path_a = pos[start : step + 1, a]  # (≤tail, 3) recent positions of body a
            line.set_data_3d(path_a[:, 0], path_a[:, 1], path_a[:, 2])
            dot.set_data_3d([pos[step, a, 0]], [pos[step, a, 1]], [pos[step, a, 2]])
        clock.set_text(f"t = {t[step]:.2f}")
        if checks:
            e_line.set_data(t[: step + 1], e_err[: step + 1])
            p_line.set_data(t[: step + 1], p_err[: step + 1])
        ax.view_init(elev=25, azim=-60 + 120 * step / max(1, len(t) - 1))  # a third of a turn over the run

    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path, fps=30)  # 1.5× the default 20 fps
