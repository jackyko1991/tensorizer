"""3D animation: each body leaves a fading trail; marker size grows with mass; the camera turns slowly to show depth."""

from pathlib import Path

import matplotlib.pyplot as plt
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
    shown as a live readout (the conservation laws the run must obey)."""
    pos = xs.cpu().numpy()  # (T, N, 3)
    t = times.cpu().numpy()  # (T,)
    mass = m.cpu().numpy()  # (N,)
    tail = max(2, int(trail * len(t)))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    half = abs(pos).max() * 1.05
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # a body holding most of the total mass is drawn as a yellow star
    colors = ["#f59e0b" if mi > 0.5 * mass.sum() else ACCENT[a % len(ACCENT)] for a, mi in enumerate(mass)]
    sizes = 4 + 8 * (mass / mass.max()) ** (1 / 3)  # radius ∝ m^(1/3)
    trails = [ax.plot([], [], [], lw=1.2, color=c, alpha=0.7)[0] for c in colors]
    dots = [ax.plot([], [], [], "o", ms=s, color=c)[0] for s, c in zip(sizes, colors)]
    clock = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    readout = ax.text2D(
        0.98, 0.97, "", transform=ax.transAxes, ha="right", va="top", fontsize=8, family="monospace", color="#374151"
    )
    e_drift = drift.cpu().numpy() if drift is not None else None
    p_total = p_norm.cpu().numpy() if p_norm is not None else None
    ax.set_title(f"{len(mass)} bodies, G = 1", fontsize=10)

    def update(step: int):
        start = max(0, step - tail)
        for a, (line, dot) in enumerate(zip(trails, dots)):
            path_a = pos[start : step + 1, a]  # (≤tail, 3) recent positions of body a
            line.set_data_3d(path_a[:, 0], path_a[:, 1], path_a[:, 2])
            dot.set_data_3d([pos[step, a, 0]], [pos[step, a, 1]], [pos[step, a, 2]])
        clock.set_text(f"t = {t[step]:.2f}")
        if e_drift is not None and p_total is not None:
            readout.set_text(f"ΔE/|E₀| = {e_drift[step]:+.1e}\n|P|      = {p_total[step]:.1e}")
        ax.view_init(elev=25, azim=-60 + 120 * step / max(1, len(t) - 1))  # a third of a turn over the run

    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path, fps=30)  # 1.5× the default 20 fps
