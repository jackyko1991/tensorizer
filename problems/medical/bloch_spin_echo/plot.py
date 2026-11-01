"""Animate isochromat arrows fanning out and refocusing in the transverse plane, next to the signal |s|(t)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import signal


def animate(
    times: torch.Tensor,
    M: torch.Tensor,
    dw: torch.Tensor,
    pulses: list[tuple[float, float, float, float]],
    T2: float,
    t2_star: float,
    t90: float,
    path: Path,
) -> Path:
    t = times.cpu().numpy()
    mag = signal(M).abs().cpu().numpy()  # (T,)
    n = M.shape[1]
    shown = np.linspace(0, n - 1, min(n, 31)).round().astype(int)  # a few isochromats to draw
    Mxy = M[:, shown, :2].cpu().numpy()  # (T, n_shown, 2)
    net = signal(M).cpu().numpy()  # (T,) complex average
    colours = plt.cm.coolwarm(np.linspace(0, 1, len(shown)))  # blue = slower, red = faster than resonance

    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(7.6, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    ax_p.set_xlim(-1.1, 1.1)
    ax_p.set_ylim(-1.1, 1.1)
    ax_p.set_aspect("equal")
    ax_p.set_xlabel("$M_x$", labelpad=0)
    ax_p.set_ylabel("$M_y$", labelpad=0)
    ax_p.add_patch(plt.Circle((0, 0), 1, fill=False, color="#9ca3af", lw=0.8, ls=":"))
    arrows = [ax_p.plot([], [], color=c, lw=1.2, alpha=0.85)[0] for c in colours]
    (net_arrow,) = ax_p.plot([], [], color="#111827", lw=3)
    title = ax_p.set_title("")

    for start, length, flip, _ in pulses:
        ax_s.axvspan(start, start + length, color="#f59e0b", alpha=0.5, lw=0)
        ax_s.text(start + length / 2, 1.05, f"{np.degrees(flip):.0f}°", ha="center", fontsize=7, color="#b45309")
    tt = t[t >= t90]
    ax_s.plot(tt, np.exp(-(tt - t90) / T2), color="#16a34a", ls="--", lw=1, label="$e^{-t/T_2}$ (echo envelope)")
    if np.isfinite(t2_star):
        ax_s.plot(tt, np.exp(-(tt - t90) / t2_star), color="#9333ea", ls=":", lw=1.2, label="$e^{-t/T_2^*}$ (FID)")
    (curve,) = ax_s.plot([], [], color="#2563eb", lw=1.6, label="$|s(t)|$")
    ax_s.set_xlim(t[0], t[-1])
    ax_s.set_ylim(0, 1.12)
    ax_s.set_xlabel("t (ms)", labelpad=0)
    ax_s.set_ylabel("net transverse signal")
    ax_s.legend(loc="center right", fontsize=7, framealpha=0.9)

    def update(step: int):
        for arrow, (mx, my) in zip(arrows, Mxy[step]):
            arrow.set_data([0, mx], [0, my])
        net_arrow.set_data([0, net[step].real], [0, net[step].imag])
        curve.set_data(t[: step + 1], mag[: step + 1])
        title.set_text(f"t = {t[step]:5.1f} ms   |s| = {mag[step]:.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 96), path, fps=12)
