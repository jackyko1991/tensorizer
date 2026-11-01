"""Left: the three bodies of the reference run (solid) and the twin (dashed). Right: log10 of their distance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

COLORS = ["#2563eb", "#dc2626", "#16a34a"]  # masses 3, 4, 5


def animate(
    times: torch.Tensor,
    xs: torch.Tensor,
    dist: torch.Tensor,
    m: torch.Tensor,
    fit: tuple,
    eject: tuple,
    path: Path,
    trail: float = 8.0,
) -> Path:
    """xs: (T, K, N, 2); dist: (T,) phase-space distance; fit = (λ, t0, t1, c); eject = (step, body) of run 0.

    trail = how many time units of path each body drags behind it.
    """
    t = times.cpu().numpy()  # (T,)
    pos = xs.cpu().numpy()  # (T, K, N, 2)
    log_d = dist.clamp_min(1e-300).log10().cpu().numpy()  # (T,)
    mass = m.cpu().numpy()
    tail = max(2, int(trail / (t[1] - t[0])))  # trail length in stored steps
    frames = frame_indices(len(t), 90)
    reach = np.maximum.accumulate(abs(pos).max(axis=(1, 2, 3)))  # (T,) farthest any body has been so far
    reach = 1.08 * np.maximum(reach, 4.0)  # zoom out only when a body flies away

    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(7.6, 4), gridspec_kw={"width_ratios": [1, 1.15]})

    # --- left: orbits ---
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])  # the view zooms out during the escape; ticks would be redrawn every frame
    ax.set_yticks([])
    ax.set_title("solid: reference   dashed: twin (+δ)", fontsize=9)
    ref_lines = [ax.plot([], [], lw=1.1, color=c)[0] for c in COLORS]
    twin_lines = [ax.plot([], [], lw=1.0, ls="--", color=c, alpha=0.45)[0] for c in COLORS]
    sizes = 4 + 2.5 * mass / mass.max() * 2
    ref_dots = [ax.plot([], [], "o", ms=s, color=c, label=f"m = {mi:g}")[0] for s, c, mi in zip(sizes, COLORS, mass)]
    twin_dots = [ax.plot([], [], "o", ms=s, mfc="none", color=c, alpha=0.6)[0] for s, c in zip(sizes, COLORS)]
    ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
    clock = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    escape_note = ax.text(0.97, 0.03, "", transform=ax.transAxes, va="bottom", ha="right", fontsize=8, color="#b45309")

    # --- right: separation of the two runs in powers of ten (straight rise = exponential growth) ---
    ax_d.set_xlim(t[0], t[-1])
    ax_d.set_ylim(log_d.min() - 0.5, max(log_d.max(), 0.0) + 0.8)
    ax_d.set_xlabel("t")
    ax_d.set_ylabel("log₁₀ |Δ|   (phase-space distance)")
    trace = ax_d.plot([], [], lw=1.1, color="#111827")[0]
    lam, t0, t1, c = fit
    if lam == lam:  # not nan
        ax_d.plot([t0, t1], [(c + lam * t0) / 2.302585, (c + lam * t1) / 2.302585], "--", color="#f59e0b", lw=2)
        ax_d.text(0.03, 0.97, f"fit: λ ≈ {lam:.2f}", transform=ax_d.transAxes, va="top", color="#b45309")
    step_ej, body_ej = eject
    if step_ej is not None:
        pair = " + ".join(f"{mi:g}" for mi in mass if mi != mass[body_ej])  # the two that stay together
        ax_d.axvline(t[step_ej], color="#9ca3af", ls=":", lw=1.5)
        ax_d.text(
            t[step_ej],
            ax_d.get_ylim()[0] + 0.3,
            f"m = {mass[body_ej]:g} escapes ",
            ha="right",
            fontsize=8,
            color="#374151",
        )

    def update(step: int):
        start = max(0, step - tail)
        for a in range(len(mass)):
            ref_lines[a].set_data(pos[start : step + 1, 0, a, 0], pos[start : step + 1, 0, a, 1])
            twin_lines[a].set_data(pos[start : step + 1, 1, a, 0], pos[start : step + 1, 1, a, 1])
            ref_dots[a].set_data([pos[step, 0, a, 0]], [pos[step, 0, a, 1]])
            twin_dots[a].set_data([pos[step, 1, a, 0]], [pos[step, 1, a, 1]])
        ax.set_xlim(-reach[step], reach[step])
        ax.set_ylim(-reach[step], reach[step])
        clock.set_text(f"t = {t[step]:.1f}")
        if step_ej is not None and step >= step_ej:
            escape_note.set_text(f"m = {mass[body_ej]:g} escapes,\nbinary {pair} left")
        trace.set_data(t[: step + 1], log_d[: step + 1])

    fig.tight_layout()
    return save_gif(fig, update, frames, path)
