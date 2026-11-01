"""3D animation of the spinning box (world frame) next to its body-frame angular velocity ω_i(t)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import world_angular_momentum

# the 6 faces of a box, as indices into the 8 corners built below
FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
FACE_COLORS = [ACCENT[0], ACCENT[0], ACCENT[1], ACCENT[1], ACCENT[2], ACCENT[2]]  # opposite faces match


def animate(sol: dict, inertia: torch.Tensor, sides: tuple[float, float, float], path: Path) -> Path:
    t = sol["times"].cpu().numpy()  # (T,)
    R = sol["R"].cpu().numpy()  # (T, 3, 3) body → world
    omega = sol["omega"].cpu().numpy()  # (T, 3) body frame
    L_world = world_angular_momentum(sol["R"][0], sol["omega"][0], inertia).cpu().numpy()  # (3,) fixed in space

    half = np.array(sides) / 2
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])  # (8, 3)
    corners = signs * half  # (8, 3) in body coordinates
    reach = np.linalg.norm(half) * 1.05

    fig = plt.figure(figsize=(7, 4))
    ax_box = fig.add_subplot(1, 2, 1, projection="3d")
    ax_w = fig.add_subplot(1, 2, 2)

    # --- left: the box in the world frame, with the fixed angular momentum ---
    for set_lim in (ax_box.set_xlim, ax_box.set_ylim, ax_box.set_zlim):
        set_lim(-reach, reach)
    ax_box.set_box_aspect((1, 1, 1))
    ax_box.set_xticklabels([])
    ax_box.set_yticklabels([])
    ax_box.set_zticklabels([])
    box = Poly3DCollection([corners[f] for f in FACES], facecolors=FACE_COLORS, edgecolor="#111827", alpha=0.85)
    ax_box.add_collection3d(box)
    L_dir = 1.3 * reach * L_world / np.linalg.norm(L_world)
    ax_box.plot([0, L_dir[0]], [0, L_dir[1]], [0, L_dir[2]], color="#111827", lw=2)
    ax_box.text(*L_dir, "L", fontsize=10)
    clock = ax_box.text2D(0.02, 0.95, "", transform=ax_box.transAxes, fontsize=9)

    # --- right: body-frame ω components ---
    labels = ["ω₁ minor", "ω₂ intermediate", "ω₃ major"]
    ax_w.set_xlim(t[0], t[-1])
    ax_w.set_ylim(omega.min() - 0.5, omega.max() + 0.5)
    ax_w.set_xlabel("t (s)")
    ax_w.set_ylabel("ω in body frame (rad/s)")
    lines = [ax_w.plot([], [], lw=1.4, color=ACCENT[i], label=labels[i])[0] for i in range(3)]
    ax_w.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)

    def update(step: int):
        world = corners @ R[step].T  # (8, 3) x^{world}_i = R_ij x^{body}_j for every corner
        box.set_verts([world[f] for f in FACES])
        for i, line in enumerate(lines):
            line.set_data(t[: step + 1], omega[: step + 1, i])
        clock.set_text(f"t = {t[step]:.2f} s")

    ax_box.view_init(elev=20, azim=-50)
    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 120), path)
