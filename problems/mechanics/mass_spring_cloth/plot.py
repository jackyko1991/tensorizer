"""3D animation of the falling cloth as a shaded surface, with its total energy beside it."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(times: torch.Tensor, traj: torch.Tensor, E: torch.Tensor, n: int, path: Path) -> Path:
    t = times.cpu().numpy()  # (T,)
    x = traj[:, 0].cpu().numpy().reshape(len(t), n, n, 3)  # (T, n, n, 3) node (r, c) → position
    energy = E.cpu().numpy()  # (T,)
    lo, hi = x.reshape(-1, 3).min(0), x.reshape(-1, 3).max(0)

    fig = plt.figure(figsize=(7, 4))
    ax_cloth = fig.add_subplot(1, 2, 1, projection="3d")
    ax_energy = fig.add_subplot(1, 2, 2)

    # --- left: the cloth ---
    ax_cloth.set_xlim(lo[0], hi[0])
    ax_cloth.set_ylim(lo[1], hi[1])
    ax_cloth.set_zlim(lo[2], hi[2])
    ax_cloth.set_box_aspect(hi - lo + 1e-9)  # true proportions
    ax_cloth.set_xlabel("x (m)", fontsize=8)
    ax_cloth.set_ylabel("y (m)", fontsize=8)
    ax_cloth.set_zlabel("z (m)", fontsize=8)
    ax_cloth.tick_params(labelsize=7)
    for axis in (ax_cloth.xaxis, ax_cloth.yaxis, ax_cloth.zaxis):
        axis.set_major_locator(plt.MaxNLocator(3))
    pins = x[0, 0, [0, n - 1]]  # (2, 3) the pinned corners never move
    ax_cloth.plot(pins[:, 0], pins[:, 1], pins[:, 2], "k^", ms=6)
    clock = ax_cloth.text2D(0.02, 0.95, "", transform=ax_cloth.transAxes, fontsize=9)
    surface = [None]  # the surface is redrawn every frame

    # --- right: total energy (only goes down with damping) ---
    ax_energy.plot(t, energy, lw=1, color="#d1d5db")
    (line,) = ax_energy.plot([], [], lw=1.6, color=ACCENT[1])
    ax_energy.set_xlim(t[0], t[-1])
    ax_energy.set_xlabel("t (s)")
    ax_energy.set_ylabel("total energy (J)")

    def update(step: int):
        if surface[0] is not None:
            surface[0].remove()
        X, Y, Z = x[step, ..., 0], x[step, ..., 1], x[step, ..., 2]
        surface[0] = ax_cloth.plot_surface(X, Y, Z, color="#93c5fd", edgecolor="#1e40af", lw=0.4, shade=True)
        line.set_data(t[: step + 1], energy[: step + 1])
        clock.set_text(f"t = {t[step]:.2f} s")

    ax_cloth.view_init(elev=15, azim=-60)
    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path)
