"""Animate the fields rolling down their potentials (e-fold counter), then their points in the (n_s, r) plane."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import tensors


def animate(view: dict, N_star: float, N_start: float, path: Path) -> Path:
    N, phi, N_end, phi0 = view["N"], view["phi"], view["N_end"], view["phi0"]
    B = len(tensors.KINDS)
    colors = [ACCENT[k] for k in range(B)]
    to_go = np.linspace(N_start, 0.0, 56)  # the clock: e-folds left before each field's end of inflation
    schedule = list(to_go) + [0.0] * 14

    def field_at(left: float) -> np.ndarray:  # φ_b at N_end,b − left, for every field, (B,)
        return np.array([np.interp(N_end[b] - left, N, phi[:, b]) for b in range(B)])

    fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(8.4, 4), gridspec_kw={"width_ratios": [1, 1]})
    phi_end = field_at(0.0)
    curves, balls = [], []
    for b, kind in enumerate(tensors.KINDS):
        lo, hi = sorted([phi0[b], phi_end[b]])
        grid = np.linspace(max(lo - 1.0, 0.0), hi + 1.0, 300)
        V = view["V"](grid)[:, b] / view["V"](np.array([phi0[b]]))[0, b]  # V / V(φ_start)
        curves.append(ax.plot(grid, V, color=colors[b], lw=1.6, label=tensors.LABELS[kind])[0])
        balls.append(ax.plot([], [], "o", ms=8, color=colors[b], mec="#111827", zorder=5)[0])
    ax.set_xlim(0, 1.05 * max(phi0.max(), phi_end.max()) + 1)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("field φ  (M_pl)")
    ax.set_ylabel("V(φ) / V(φ_start)")
    ax.legend(fontsize=7, loc="lower right")
    clock = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title("each field rolls downhill; ● = now", fontsize=9)

    # observations: Planck n_s bands (1σ, 2σ) and the BICEP/Keck upper limit on r
    m, s = tensors.NS_MEAN, tensors.NS_SIGMA
    ax_p.axvspan(m - 2 * s, m + 2 * s, ymax=1, color="#93c5fd", alpha=0.35, lw=0)
    ax_p.axvspan(m - s, m + s, color="#2563eb", alpha=0.25, lw=0, label="Planck 2018 n_s (1σ, 2σ)")
    ax_p.axhspan(tensors.R_MAX, 1.0, color="#fecaca", alpha=0.45, lw=0, label="r > 0.036 excluded (BK18)")
    ax_p.axhline(tensors.R_MAX, color=ACCENT[1], lw=1.2, ls="--")
    ax_p.set_yscale("log")
    ax_p.set_xlim(0.935, 0.985)
    ax_p.set_ylim(1e-3, 0.6)
    ax_p.set_xlabel("spectral index n_s")
    ax_p.set_ylabel("tensor-to-scalar ratio r")
    tracks = [ax_p.plot([], [], "-", color=colors[b], lw=2.5, alpha=0.8)[0] for b in range(B)]
    dots = [ax_p.plot([], [], "o", ms=7, color=colors[b], mec="#111827", zorder=5)[0] for b in range(B)]
    sr = [ax_p.plot([], [], "x", ms=6, color=colors[b], zorder=6)[0] for b in range(B)]
    ax_p.legend(fontsize=7, loc="lower left")
    verdict = ax_p.text(
        0.97,
        0.97,
        "",
        transform=ax_p.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax_p.set_title("predictions at the pivot, N_* = 60 → 50", fontsize=9)
    Ng, ns_t, r_t = view["N_grid"], view["ns_track"], view["r_track"]  # (S,), (S, B), (S, B)

    def update(i: int):
        left = schedule[i]
        now = field_at(left)
        V_now = view["V"](now)[np.arange(B), np.arange(B)] / view["V"](phi0)[np.arange(B), np.arange(B)]
        for b in range(B):
            balls[b].set_data([now[b]], [V_now[b]])
        phase = "inflating" if left > 0 else "inflation over (ε = 1)"
        window = "  ← pivot scales leave the horizon" if 50 <= left <= 60 else ""
        clock.set_text(f"e-folds to go: {left:5.1f}{window}\n{phase}")
        upto = Ng >= max(left, 50.0)  # tracks drawn from N_* = 60 down to the current clock
        for b in range(B):
            if left <= 60:
                tracks[b].set_data(ns_t[upto, b], r_t[upto, b])
                k = int(np.argmin(np.abs(Ng - max(left, 50.0))))
                dots[b].set_data([ns_t[k, b]], [r_t[k, b]])
        if i >= len(to_go):  # final frames: the chosen N_*, slow-roll crosses and the verdict
            for b in range(B):
                dots[b].set_data([view["ns"][b]], [view["r"][b]])
                sr[b].set_data([view["ns_V"][b]], [view["r_V"][b]])
            lines = [
                f"{tensors.LABELS[k]}: {'inside' if view['ok'][b] else 'disfavoured'}"
                for b, k in enumerate(tensors.KINDS)
            ]
            verdict.set_text(f"N_* = {N_star:.0f}   (● numerical, × slow roll)\n" + "\n".join(lines))

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(schedule), 70), path)
