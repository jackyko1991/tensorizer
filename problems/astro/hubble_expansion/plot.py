"""Animate the expanding galaxy grid (seen from one observer) next to the velocity–distance plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import KMS_PER_MPC_GYR


def animate(sol: dict, observer: int, centre: int, path: Path, arrow: float = 0.3) -> Path:
    """Velocity arrows are drawn as arrow · v / H, so their length is a fixed fraction of the distance to ★."""
    t, a, H = (sol[k].cpu().numpy() for k in ("t", "a", "H"))
    x = sol["x"].cpu().numpy()  # (T, G, 2)
    v_rel = sol["v_rel"].cpu().numpy()
    d_obs = np.linalg.norm(sol["x_rel"].cpu().numpy(), axis=-1)  # (T, G) distance from the observer
    s_obs = np.linalg.norm(v_rel, axis=-1) * KMS_PER_MPC_GYR / 1e3  # (T, G) speed in 1000 km/s
    x_c = x - x[:, centre : centre + 1]
    d_c = np.linalg.norm(x_c, axis=-1)
    s_c = np.linalg.norm(sol["v"].cpu().numpy() - sol["v"].cpu().numpy()[:, centre : centre + 1], axis=-1)
    s_c = s_c * KMS_PER_MPC_GYR / 1e3
    emitter, photon, z = sol["emitter"].cpu().numpy(), sol["photon"].cpu().numpy(), sol["z"].cpu().numpy()

    fig, (ax, ax_v) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    reach = 1.05 * max(np.abs(x[-1]).max(), np.abs(emitter[-1]).max())
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (Mpc)")
    ax.set_ylabel("y (Mpc)")
    galaxies = ax.plot([], [], "o", ms=3.5, color="#6b7280")[0]
    arrows = ax.quiver(
        x[0, :, 0], x[0, :, 1], 0 * v_rel[0, :, 0], 0 * v_rel[0, :, 1],
        angles="xy", scale_units="xy", scale=1.0, width=0.004, color=ACCENT[1], alpha=0.7,
    )  # fmt: skip
    me = ax.plot([], [], "*", ms=13, color=ACCENT[0], zorder=5)[0]
    src = ax.plot([], [], "D", ms=6, color="#f59e0b", zorder=5)[0]
    trail = ax.plot([], [], "-", lw=1.5, color="#f59e0b")[0]
    light = ax.plot([], [], "o", ms=5, color="#b45309", zorder=6)[0]
    label = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_title("★ observer   arrows ∝ velocity seen by ★   ◆ photon source", fontsize=8)

    d_max, s_max = 1.05 * d_obs.max(), 1.05 * max(s_obs.max(), s_c.max())
    ax_v.set_xlim(0, d_max)
    ax_v.set_ylim(0, s_max)
    ax_v.set_xlabel("distance d (Mpc)")
    ax_v.set_ylabel("recession speed v (1000 km/s)")
    pts_c = ax_v.plot([], [], "o", ms=6, mfc="none", color="#9ca3af", label="seen from the centre")[0]
    pts_o = ax_v.plot([], [], "o", ms=3, color=ACCENT[0], label="seen from ★")[0]
    law = ax_v.plot([], [], "--", color=ACCENT[1], lw=1.2, label="v = H d")[0]
    ax_v.legend(fontsize=8, loc="upper left")
    title = ax_v.set_title("", fontsize=9)

    def update(step: int):
        galaxies.set_data(x[step, :, 0], x[step, :, 1])
        arrows.set_offsets(x[step])
        arrows.set_UVC(arrow * v_rel[step, :, 0] / H[step], arrow * v_rel[step, :, 1] / H[step])
        me.set_data([x[step, observer, 0]], [x[step, observer, 1]])
        src.set_data([emitter[step, 0]], [emitter[step, 1]])
        path_now = a[step] / a[: step + 1, None] * photon[: step + 1]  # light's path so far, stretched to today
        trail.set_data(path_now[:, 0], path_now[:, 1])
        light.set_data([photon[step, 0]], [photon[step, 1]])
        label.set_text(f"t = {t[step]:.1f} Gyr   a = {a[step]:.2f}\nphoton stretched: 1 + z = {1 + z[step]:.2f}")
        pts_c.set_data(d_c[step], s_c[step])
        pts_o.set_data(d_obs[step], s_obs[step])
        H_kms = H[step] * KMS_PER_MPC_GYR
        law.set_data([0, d_max], [0, H_kms * d_max / 1e3])
        title.set_text(f"Hubble law: H = ȧ/a = {H_kms:.0f} km/s/Mpc")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
