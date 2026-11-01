"""Animate the falling ring: the whole fall, a zoom on its shape (exact vs tidal tensor), and its size vs r."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, save_gif

from .equation import tidal_tensor


def frames_by_radius(r: np.ndarray, n_frames: int) -> list[int]:
    """Steps where r crosses evenly spaced values of log r: the fall speeds up, the animation should not."""
    from problems._common.plotting import frame_indices

    n_frames = len(frame_indices(len(r), n_frames))  # respects fast mode
    targets = np.geomspace(r[0], r[-1], n_frames)
    order = np.argsort(r)  # r decreases with time
    steps = np.interp(targets, r[order], order)
    return sorted({int(round(s)) for s in steps})


def animate(sol: dict, GM: float, r_t: float, path: Path) -> Path:
    centre = sol["centre"].cpu().numpy()  # (T, 3)
    ring = sol["ring"].cpu().numpy()  # (T, n, 3)
    xi_e, xi_l = sol["xi_exact"].cpu().numpy(), sol["xi_tidal"].cpu().numpy()
    r = np.linalg.norm(centre, axis=-1)
    s0 = np.linalg.norm(xi_e[0], axis=-1).max()
    along_e, across_e = np.abs(xi_e[..., 0]).max(-1), np.abs(xi_e[..., 1]).max(-1)
    along_l, across_l = np.abs(xi_l[..., 0]).max(-1), np.abs(xi_l[..., 1]).max(-1)
    T = tidal_tensor(sol["centre"], GM).cpu().numpy()  # (T, 3, 3)

    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
    ax = fig.add_subplot(gs[0, :])
    ax_z = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])

    # --- the whole fall, true scale ---
    ax.set_xlim(-3, r[0] + 2)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(plt.Circle((0, 0), 2 * GM, color="black", zorder=4))
    ax.add_patch(plt.Circle((0, 0), r_t, fill=False, ls="--", color="#f59e0b", lw=1.2))
    ax.text(r_t, 1.2, "tidal radius r_t", color="#b45309", fontsize=8)
    ax.text(2.5, -3.2, "horizon r_s = 2GM/c²", fontsize=8)
    ax.set_xlabel("x  (GM/c²)")
    blob = ax.fill([], [], color=ACCENT[0], ec=ACCENT[0], lw=2.5, zorder=5)[0]
    path_line = ax.plot([], [], color="#9ca3af", lw=0.8)[0]

    # --- zoom on the deformation, in the ring's own (falling) frame ---
    phi = np.linspace(0, 2 * np.pi, 100)
    ax_z.plot(s0 * np.cos(phi), s0 * np.sin(phi), "--", color="#9ca3af", lw=1, label="start")
    exact = ax_z.plot([], [], "o", ms=3, color=ACCENT[0], label="exact gravity")[0]
    linear = ax_z.plot([], [], "-", color=ACCENT[1], lw=1.2, label="tidal tensor T·ξ")[0]
    ax_z.set_aspect("equal", adjustable="box")
    ax_z.set_xlabel("ξ_x  (radial, black hole ←)")
    ax_z.set_ylabel("ξ_y")
    ax_z.legend(fontsize=7, loc="upper right")
    title_z = ax_z.set_title("", fontsize=8)

    # --- half-lengths against distance ---
    ax_s.loglog(r, along_e / s0, color=ACCENT[0], lw=1, alpha=0.35)
    ax_s.loglog(r, across_e / s0, color=ACCENT[2], lw=1, alpha=0.35)
    ax_s.set_xlim(r[0] * 1.05, r[-1] * 0.95)  # r decreases to the right, as time goes on
    ax_s.axvline(r_t, color="#f59e0b", ls="--", lw=1)
    run_along = ax_s.loglog([], [], color=ACCENT[0], lw=2, label="radial (stretch)")[0]
    run_across = ax_s.loglog([], [], color=ACCENT[2], lw=2, label="transverse (squeeze)")[0]
    run_lin = ax_s.loglog([], [], ":", color=ACCENT[1], lw=1.5, label="tidal tensor")[0]
    ax_s.loglog([], [], ":", color=ACCENT[1], lw=1.5)
    ax_s.set_xlabel("distance r  (GM/c²)")
    ax_s.set_ylabel("half-length / initial radius")
    ax_s.legend(fontsize=7, loc="upper left")
    ticks = [v for v in (100, 50, 40, 20, 10, 5, 3, 2) if r[-1] <= v <= r[0]]
    ax_s.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax_s.minorticks_off()

    def update(step: int):
        outline = ring[step, :, :2]
        blob.set_xy(outline)
        path_line.set_data(centre[: step + 1, 0], centre[: step + 1, 1])
        exact.set_data(xi_e[step, :, 0], xi_e[step, :, 1])
        loop = np.vstack([xi_l[step], xi_l[step, :1]])
        linear.set_data(loop[:, 0], loop[:, 1])
        reach = 1.15 * max(s0, np.abs(xi_e[step, :, :2]).max())
        ax_z.set_xlim(-reach, reach)
        ax_z.set_ylim(-reach, reach)
        ev = np.linalg.eigvalsh(T[step]) * r[step] ** 3 / GM
        title_z.set_text(f"r = {r[step]:.1f}:  eig(T)·r³/GM = {ev[2]:+.2f}, {ev[1]:+.2f}, {ev[0]:+.2f}")
        run_along.set_data(r[: step + 1], along_e[: step + 1] / s0)
        run_across.set_data(r[: step + 1], across_e[: step + 1] / s0)
        stacked = np.concatenate([along_l[: step + 1], [np.nan], across_l[: step + 1]]) / s0
        run_lin.set_data(np.concatenate([r[: step + 1], [np.nan], r[: step + 1]]), stacked)

    fig.tight_layout()
    return save_gif(fig, update, frames_by_radius(r, 90), path)
