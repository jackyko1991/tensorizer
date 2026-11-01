"""Animate the ozone layer forming after sunrise: profiles, time series at three heights and the two time scales."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif

BLUE, RED, GREY, AMBER, DARK, GREEN = "#2563eb", "#dc2626", "#9ca3af", "#f59e0b", "#374151", "#16a34a"


def _when(t: float) -> str:
    for unit, s in (("yr", 3.156e7), ("days", 86400), ("h", 3600), ("min", 60), ("s", 1)):
        if t >= s:
            return f"{t / s:.3g} {unit}"
    return f"{t:.1e} s"


def _decades(ax, axis: str, lo: int, hi: int, step: int = 2):
    ticks = [10.0**k for k in range(lo, hi + 1, step)]
    labels = [f"1e{k}" for k in range(lo, hi + 1, step)]  # plain labels: no mathtext to typeset per frame
    if axis == "x":
        ax.set_xscale("log")
        ax.set_xticks(ticks, labels)
        ax.set_xlim(ticks[0], ticks[-1])
    else:
        ax.set_yscale("log")
        ax.set_yticks(ticks, labels)
        ax.set_ylim(ticks[0], ticks[-1])
    ax.minorticks_off()


def animate(times, z, c, O3_ss, fast_tau, slow_tau, path: Path) -> Path:
    """times (T,) s; z (A,) km; c (T, A, 3) of (O, O2, O3) in cm⁻³; steady state and time scales (A,)."""
    t = times.numpy()
    zs = z.cpu().numpy()
    C = c.cpu().numpy()
    ss, tf, tx = O3_ss.cpu().numpy(), fast_tau.cpu().numpy(), slow_tau.cpu().numpy()
    tt = np.maximum(t, 1e-5)  # t = 0 drawn at the left edge of the log axis
    floor = 1e2

    fig, (ax_z, ax_t, ax_l) = plt.subplots(1, 3, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    fig.subplots_adjust(left=0.065, right=0.98, top=0.88, bottom=0.13, wspace=0.32)

    # --- altitude profiles ---
    ax_z.plot(ss, zs, color=DARK, lw=1.2, ls=(0, (4, 3)), label="O₃ exact steady state")
    (l_o3,) = ax_z.plot([], [], color=BLUE, lw=2, label="O₃")
    (l_o,) = ax_z.plot([], [], color=RED, lw=1.5, label="O")
    _decades(ax_z, "x", 2, 14)
    ax_z.set_ylim(zs[0], zs[-1])
    ax_z.set_xlabel("number density (cm⁻³)", labelpad=1)
    ax_z.set_ylabel("altitude (km)", labelpad=1)
    ax_z.legend(fontsize=7, loc="upper left", frameon=False)
    title = ax_z.set_title("", fontsize=9)

    # --- time series at three heights ---
    picks = [int(np.abs(zs - h).argmin()) for h in (20, 30, 45)]
    shades = ["#93c5fd", BLUE, "#1e3a8a"]
    for a, col in zip(picks, shades):
        ax_t.plot(tt, np.maximum(C[:, a, 2], floor), color=col, lw=1.8, label=f"O₃ {zs[a]:.0f} km")
        ax_t.plot(tt, np.maximum(C[:, a, 0], floor), color=col, lw=1, ls=(0, (3, 2)))
        ax_t.plot([tf[a]], [floor * 3], "v", color=AMBER, ms=6, mec=DARK, mew=0.5)
        if tx[a] < t[-1]:
            ax_t.plot([tx[a]], [floor * 3], "v", color=GREEN, ms=6, mec=DARK, mew=0.5)
    _decades(ax_t, "x", -4, int(np.ceil(np.log10(t[-1]))))
    _decades(ax_t, "y", 2, 14)
    ax_t.set_xlabel("time since sunrise (s)", labelpad=1)
    ax_t.set_title("solid O₃, dashed O; ▼ amber τ_fast, ▼ green τ_Oₓ", fontsize=9)
    ax_t.legend(fontsize=7, loc="upper left", frameon=False)
    (cursor,) = ax_t.plot([1e-4, 1e-4], [1e2, 1e14], color=GREY, lw=1)

    # --- time scales: fast O ⇄ O₃ exchange, slow odd-oxygen relaxation ---
    ax_l.plot(tf, zs, color=AMBER, lw=2, label="τ_fast: O ⇄ O₃")
    ax_l.plot(tx, zs, color=GREEN, lw=2, label="τ_Oₓ: odd oxygen")
    for label, sec, y in (("1 min", 60, 1), ("1 day", 86400, 1), ("1 yr", 3.156e7, 4)):
        ax_l.axvline(sec, color="#e5e7eb", lw=1)
        ax_l.text(sec, zs[-1] - y, label, fontsize=7, color="#6b7280", ha="center", va="top")
    _decades(ax_l, "x", -4, 12, step=4)
    ax_l.set_ylim(zs[0], zs[-1])
    ax_l.set_xlabel("time scale (s)", labelpad=1)
    ax_l.legend(fontsize=7, loc="lower left", frameon=False)
    ax_l.set_title("left of the cursor: settled", fontsize=9)
    (cursor_l,) = ax_l.plot([1e-4, 1e-4], [zs[0], zs[-1]], color=GREY, lw=1)
    fig.suptitle("Chapman ozone layer: dc/dt = S·r(c), 4 reactions × 3 species at every altitude", fontsize=10)

    def update(k: int):
        l_o3.set_data(np.maximum(C[k, :, 2], floor), zs)
        l_o.set_data(np.maximum(C[k, :, 0], floor), zs)
        title.set_text(f"profiles, t = {_when(t[k])}" if k else "profiles at sunrise (all O₂)")
        cursor.set_data([tt[k], tt[k]], [1e2, 1e14])
        cursor_l.set_data([tt[k], tt[k]], [zs[0], zs[-1]])

    return save_gif(fig, update, frame_indices(len(t), 60), path, fps=8)
