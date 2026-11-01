"""Animate a small black hole evaporating: shrinking horizon, hotter and bluer spectrum, final burst."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import planck, wien_peak


def animate(times: torch.Tensor, masses: torch.Tensor, T: torch.Tensor, P: torch.Tensor, lam: torch.Tensor,
            t_life: float, path: Path) -> Path:  # fmt: skip
    """times, masses, T, P: (S,) histories of the animated hole; lam: wavelength grid (L,)."""
    frames = frame_indices(len(times), 85)
    steps = torch.tensor(frames)
    spectra = planck(lam, T[steps]).cpu().numpy()  # (F, L): one Planck curve per frame, by broadcasting
    t = times.cpu().numpy() / t_life
    m = (masses / masses[0]).cpu().numpy()
    T_np, P_np, lam_np = T.cpu().numpy(), P.cpu().numpy(), lam.cpu().numpy()
    peaks = wien_peak(T).cpu().numpy()
    burst = [frames[-1]] * 12  # hold the end with a flash

    fig = plt.figure(figsize=(8, 4.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 1])

    # --- the hole itself: radius ∝ M, glow colour from the temperature ---
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-2.1, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    cmap = plt.cm.inferno
    logT = np.log10(T_np)
    shade = lambda k: cmap(0.25 + 0.7 * (logT[k] - logT[0]) / max(1e-9, logT[-1] - logT[0]))  # noqa: E731
    glow = plt.Circle((0, 0), 1.25, color=shade(0), alpha=0.5)
    hole = plt.Circle((0, 0), 1.0, color="black")
    ax.add_patch(glow)
    ax.add_patch(hole)
    rays = [ax.plot([], [], color="#f59e0b", lw=2)[0] for _ in range(16)]
    info = ax.text(0, -2.05, "", ha="center", va="bottom", fontsize=8, family="monospace")
    ax.set_title("black hole (drawn with size ∝ M^0.35)", fontsize=9)

    # --- spectra ---
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.set_xlim(lam_np[0], lam_np[-1])
    top = spectra.max()
    ax_s.set_ylim(top * 1e-14, top * 5)
    ax_s.set_xlabel("wavelength λ (m)", fontsize=8)
    ax_s.set_ylabel("B_λ (W sr⁻¹ m⁻³)", fontsize=8)
    ax_s.tick_params(labelsize=7)
    ghosts = [ax_s.plot(lam_np, s, color="#d1d5db", lw=0.6)[0] for s in spectra]
    for g in ghosts:
        g.set_visible(False)
    now = ax_s.plot([], [], color=ACCENT[1], lw=2)[0]
    peak = ax_s.axvline(peaks[0], color=ACCENT[0], ls="--", lw=1)
    ax_s.set_title("Planck spectrum at T_H, dashed: Wien peak b/T_H", fontsize=8)

    # --- mass history ---
    s = np.linspace(0, 1, 400)
    ax_m.plot(s, (1 - s) ** (1 / 3), color="#9ca3af", lw=1, label="(1 − t/t_life)^{1/3}")
    run = ax_m.plot([], [], "o", ms=2.5, color=ACCENT[0], label="RK4")[0]
    ax_m.set_xlim(0, 1.02)
    ax_m.set_ylim(0, 1.05)
    ax_m.set_xlabel("t / t_life", fontsize=8)
    ax_m.set_ylabel("M / M₀", fontsize=8)
    ax_m.tick_params(labelsize=7)
    ax_m.legend(fontsize=7, loc="lower left")

    shown = {"n": 0}

    def update(step: int):
        k = frames.index(step)
        final = step == frames[-1] and shown["n"] >= len(frames)
        shown["n"] += 1
        r = m[step] ** 0.35  # gentle compression so the hole stays visible to the end
        hole.set_radius(r)
        glow.set_radius(r * 1.25 + (0.6 if final else 0.0))
        glow.set_color(shade(step))
        glow.set_alpha(0.9 if final else 0.5)
        angles = np.linspace(0, 2 * np.pi, len(rays), endpoint=False)
        reach = 1.5 if final else r * 1.25 + 0.25 * (1 - m[step])
        for ray, a in zip(rays, angles):
            ray.set_data([r * 1.3 * np.cos(a), reach * np.cos(a)], [r * 1.3 * np.sin(a), reach * np.sin(a)])
            ray.set_visible(final or reach > r * 1.35)
        for j, g in enumerate(ghosts):
            g.set_visible(j < k and j % 6 == 0)  # every 6th earlier spectrum
        now.set_data(lam_np, spectra[k])
        peak.set_xdata([peaks[step], peaks[step]])
        run.set_data(t[: step + 1], m[: step + 1])
        left = (1 - t[step]) * t_life
        info.set_text(
            f"M = {masses[step]:.2e} kg   T_H = {T_np[step]:.2e} K\nP = {P_np[step]:.2e} W   time left {left:.2e} s"
            + ("\nfinal burst!" if final else "")
        )

    fig.tight_layout()
    return save_gif(fig, update, frames + burst, path)
