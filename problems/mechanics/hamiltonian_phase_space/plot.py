"""Animate the phase portrait (orbits, separatrix, Liouville blob) next to energy error and blob area per integrator."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif
from problems._common.runtime import fast_mode

COLORS = {"leapfrog": ACCENT[0], "RK4": ACCENT[2], "Euler": ACCENT[1]}


def wrap(q: np.ndarray) -> np.ndarray:
    """Angle into [−π, π), with NaN where an orbit jumps across the edge (so no line is drawn across)."""
    w = (q + math.pi) % (2 * math.pi) - math.pi
    jumps = np.abs(np.diff(w, axis=0)) > math.pi
    w[1:][jumps] = np.nan
    return w


def animate(sol: dict, energy: dict, area: dict, scale: float, p_sep: float, system: str, path: Path) -> Path:
    t = sol["times"].cpu().numpy()  # (T,)
    leap = sol["traj"]["leapfrog"]
    orbits = leap["orbits"].cpu().numpy()  # (T, B, 2)
    blob = leap["blob"].cpu().numpy()  # (T, K, 2)
    pendulum = system == "pendulum"
    q_orb = wrap(orbits[..., 0]) if pendulum else orbits[..., 0]
    p_orb = orbits[..., 1]
    err = {k: np.abs((e - e[0]).cpu().numpy()) / scale + 1e-16 for k, e in energy.items()}  # (T,) each
    ratio = {k: (a / a[0]).cpu().numpy() for k, a in area.items()}  # (T,) each
    for r in ratio.values():  # once a blob leaves the scale (Euler) its polygon is too stretched to measure
        off = np.nonzero(np.abs(r - 1) > 0.5)[0]
        if len(off):
            r[off[0] + 1 :] = np.nan

    fig = plt.figure(figsize=(8, 4.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax_e = fig.add_subplot(grid[0, 1])
    ax_a = fig.add_subplot(grid[1, 1], sharex=ax_e)

    # --- left: phase plane (q, p) ---
    q_lim = math.pi if pendulum else 1.1 * np.nanmax(np.abs(q_orb))
    ax.set_xlim(-q_lim, q_lim)
    ax.set_ylim(-1.65 * p_sep, 1.65 * p_sep)
    ax.set_xlabel("q (rad)")
    ax.set_ylabel("p (kg·m²/s)")
    if pendulum:
        qs = np.linspace(-math.pi, math.pi, 200)
        ps = p_sep * np.cos(qs / 2)  # separatrix p = ±2mL√(gL) cos(q/2)
        for sign in (1, -1):
            ax.plot(qs, sign * ps, color="#f59e0b", lw=2.2, label="separatrix $H = mgL$" if sign > 0 else None)
    lines = [ax.plot([], [], lw=0.9, color="#6b7280")[0] for _ in range(orbits.shape[1])]
    heads = ax.plot([], [], "o", ms=3, color="#374151")[0]
    ax.fill(blob[0, :, 0], blob[0, :, 1], fill=False, ls="--", lw=0.8, color=ACCENT[0])
    patch = ax.fill(blob[0, :, 0], blob[0, :, 1], color=ACCENT[0], alpha=0.55, label="blob (leapfrog)")[0]
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_title("phase space: libration, rotation, Liouville blob", fontsize=9)

    # --- right top: energy error of the test orbit ---
    ax_e.set_yscale("log")
    ax_e.set_xlim(t[0], t[-1])
    ax_e.set_ylim(1e-9, 1e2)
    ax_e.set_yticks([1e-9, 1e-6, 1e-3, 1e0])
    ax_e.minorticks_off()
    ax_e.set_ylabel("|ΔE| / mgL", fontsize=8)
    e_lines = {k: ax_e.plot([], [], lw=0.9, color=c, label=k)[0] for k, c in COLORS.items()}
    ax_e.legend(loc="upper left", fontsize=7, ncol=3)

    # --- right bottom: blob area ---
    ax_a.set_ylim(0.9, 1.5)
    ax_a.axhline(1.0, color="#9ca3af", lw=0.8, ls="--")
    ax_a.set_ylabel("area A / A₀", fontsize=8)
    ax_a.set_xlabel("t (s)")
    a_lines = {k: ax_a.plot([], [], lw=1.2, color=c, zorder=3 if k == "leapfrog" else 2)[0] for k, c in COLORS.items()}
    ax_a.text(
        0.98,
        0.92,
        "Euler: area keeps growing ↑",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=COLORS["Euler"],
    )
    plt.setp(ax_e.get_xticklabels(), visible=False)

    def update(step: int):
        for b, line in enumerate(lines):
            line.set_data(q_orb[: step + 1, b], p_orb[: step + 1, b])
        heads.set_data(q_orb[step], p_orb[step])
        patch.set_xy(blob[step])
        for k in COLORS:
            e_lines[k].set_data(t[: step + 1], err[k][: step + 1])
            a_lines[k].set_data(t[: step + 1], ratio[k][: step + 1])
        clock.set_text(f"t = {t[step]:.1f} s")

    fig.tight_layout()
    # 240 frames at 15 fps (16 s): about real time for the default 30 s, so each orbit can be followed
    n = 240 if not fast_mode() else 5
    frames = sorted({round(k * (len(t) - 1) / (n - 1)) for k in range(n)}) if len(t) > n else frame_indices(len(t))
    return save_gif(fig, update, frames, path, fps=15)
