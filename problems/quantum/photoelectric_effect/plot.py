"""Sweep the light frequency: electron energies, retarding-voltage curves and the line eV_s = hν − φ."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import equation

AMBER, GREY = "#f59e0b", "#9ca3af"
INT_COLORS = [ACCENT[5], ACCENT[0], ACCENT[3]]


def animate(nu, sim, phi, intensities, millikan, t_wave, path: Path) -> Path:
    nu_np = nu.numpy()
    K = sim["K"].numpy()  # (I, F, N), −inf where no electron
    current = sim["current"].numpy()  # (I, F, V)
    volts = sim["voltages"].numpy()
    V_stop = sim["V_stop"].numpy()  # (I, F)
    nu_m, V_m, fit = (t.numpy() if hasattr(t, "numpy") else t for t in millikan)
    exact = equation.stopping_potential(nu, phi).numpy()
    k_top = max(float(exact.max()), 0.5)

    fig, (ax_k, ax_iv, ax_line) = plt.subplots(1, 3, figsize=(10.5, 4))

    # --- electron kinetic energies at intensity 1
    bins = np.linspace(0, k_top + 0.2, 31)
    ax_k.set_xlim(bins[0], bins[-1])
    ax_k.set_xlabel("kinetic energy $K$ (eV)")
    ax_k.set_ylabel("electrons per bin")
    kmax_line = ax_k.axvline(0, color=AMBER, lw=2, label="$h\\nu - \\phi$")
    ax_k.legend(fontsize=8, loc="upper right")
    none_text = ax_k.text(0.5, 0.5, "", transform=ax_k.transAxes, ha="center", fontsize=10, color=ACCENT[1])
    peak = max(1, int(np.isfinite(K[1]).sum(axis=-1).max() / 30 * 1.6))
    ax_k.set_ylim(0, peak)
    bars = ax_k.bar(bins[:-1], np.zeros(len(bins) - 1), width=np.diff(bins), align="edge", color=ACCENT[0], alpha=0.75)

    # --- retarding-voltage curves for 3 intensities
    ax_iv.set_xlim(volts[0], volts[-1])
    ax_iv.set_ylim(0, current.max() * 1.08 + 1e-9)
    ax_iv.set_xlabel("retarding voltage $V$ (V)")
    ax_iv.set_ylabel("photocurrent (photons → electrons)")
    iv_lines = [
        ax_iv.step([], [], where="post", color=c, lw=1.8, label=f"intensity × {float(s):g}")[0]
        for c, s in zip(INT_COLORS, intensities)
    ]
    vs_line = ax_iv.axvline(0, color=AMBER, ls=(0, (4, 3)), lw=1.5, label="$V_s$ (same for all)")
    ax_iv.legend(fontsize=7, loc="upper right")

    # --- stopping potential against frequency
    ax_line.set_xlim(nu_np[0], nu_np[-1])
    ax_line.set_ylim(-0.1, max(exact.max(), V_m.max()) + 0.6)
    ax_line.set_xlabel("frequency $\\nu$ ($10^{14}$ Hz)")
    ax_line.set_ylabel("stopping potential $V_s$ (V)")
    ax_line.plot(nu_np, exact, color=GREY, lw=1, ls=(0, (4, 3)), label=f"$hν-φ$, φ = {phi:.2f} eV")
    for c, s in zip(INT_COLORS, intensities):  # wave picture: V_s set by intensity, any ν
        ax_line.axhline(0.6 * float(s), color=c, lw=0.8, ls=":", alpha=0.8)
    ax_line.text(nu_np[-1] - 0.1, 1.25, "wave picture:\n$V_s \\propto$ intensity", ha="right", fontsize=7, color=GREY)
    xs = np.array([nu_np[0], nu_np[-1]])
    ax_line.plot(
        xs,
        fit["slope"] * xs + fit["intercept"],
        color=ACCENT[1],
        lw=1.2,
        label=f"fit: h = {fit['slope'] * 1e-14 * equation.E_CHARGE:.3e} J s",
    )
    ax_line.plot(nu_m, V_m, "s", color=ACCENT[1], ms=6, label="Millikan 1916, Na")
    sim_dots = [ax_line.plot([], [], "o", color=c, ms=ms)[0] for c, ms in zip(INT_COLORS, [6, 4, 2])]
    ax_line.legend(fontsize=7, loc="upper left")
    footer = fig.text(0.5, 0.01, "", ha="center", fontsize=9)

    def update(f: int):
        nu_f = nu_np[f]
        k = K[1, f][np.isfinite(K[1, f])]
        for bar, height in zip(bars, np.histogram(k, bins=bins)[0]):
            bar.set_height(height)
        kmax_line.set_xdata([exact[f]] * 2)
        none_text.set_text("" if len(k) else "below threshold:\nno electrons at any intensity")
        ax_k.set_title(f"ν = {nu_f:.2f}×10¹⁴ Hz: {len(k)} electrons", fontsize=9)
        for line, c in zip(iv_lines, current[:, f]):
            line.set_data(volts, c)
        vs_line.set_xdata([V_stop[1, f]] * 2)
        ax_iv.set_title(f"$V_s$ = {V_stop[1, f]:.2f} V   (exact {exact[f]:.2f} V)", fontsize=9)
        for dots, Vs in zip(sim_dots, V_stop):
            dots.set_data(nu_np[: f + 1], Vs[: f + 1])
        ax_line.set_title("$eV_s = h\\nu - \\phi$: slope $h/e$", fontsize=9)
        footer.set_text(
            f"λ = {2997.92458 / nu_f:.0f} nm,  hν = {equation.H_OVER_E * nu_f:.2f} eV,  φ = {phi:.2f} eV.   "
            f"Wave picture: wait {t_wave / 60:.0f} min to collect φ — observed: no delay (< 3 ns)"
        )

    update(0)  # titles in place before the layout is computed
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    frames = frame_indices(len(nu_np), 70) + [len(nu_np) - 1] * 10
    return save_gif(fig, update, frames, path, fps=12)
