"""Animate the light elements forming as the universe cools, then the Schramm plot of the yields against η."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .tensors import MASS_NUMBER, OBSERVED, SPECIES

COLOURS = ["#6b7280", "#111827", ACCENT[0], ACCENT[5], ACCENT[3], ACCENT[1], ACCENT[2], ACCENT[4]]


def animate(res: dict, eta_band: tuple[float, float], path: Path, n_grow: int = 50, n_sweep: int = 30) -> Path:
    """Left: mass fractions X_i = A_i Y_i against T for the last η of the batch (the marked value). Right: the
    final yields of the whole batch against η, revealed from low to high η, with the observed bands."""
    T, t = res["T"].cpu().numpy(), res["t"].cpu().numpy()
    X = (res["Y"][:, -1, :] * MASS_NUMBER.to(res["Y"])).cpu().numpy()  # (S, N) for the marked η
    etas = res["etas"].cpu().numpy()
    order = np.argsort(etas[:-1])
    eta = etas[:-1][order]
    curves = {k: res[k].cpu().numpy()[:-1][order] for k in ("Yp", "DH", "He3H", "LiH")}
    mark = {k: float(res[k][-1]) for k in ("Yp", "DH", "He3H", "LiH")}

    fig = plt.figure(figsize=(9.6, 5.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.12, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(T[0], T[-1])
    ax.set_ylim(1e-12, 2)
    ax.set_xlabel("photon temperature T (MeV)   →  time")
    ax.set_ylabel("mass fraction  $X_i = A_i Y_i$")
    lines = [ax.plot([], [], lw=1.8, color=c, label=s)[0] for s, c in zip(SPECIES, COLOURS)]
    ax.legend(fontsize=7, ncol=2, loc="center left")
    clock = ax.text(0.97, 0.97, "", transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox={"fc": "white", "ec": "#d1d5db"})  # fmt: skip
    ax.set_title(f"η = {etas[-1] * 1e10:.2f} × 10⁻¹⁰", fontsize=10)

    ax_y, ax_d, ax_l = (fig.add_subplot(gs[k, 1]) for k in range(3))
    specs = [
        (ax_y, [("Yp", ACCENT[1], "$Y_p$")], "Yp", (0.20, 0.27), False),
        (ax_d, [("DH", ACCENT[0], "D/H"), ("He3H", ACCENT[3], "³He/H")], "DH", (3e-6, 1e-3), True),
        (ax_l, [("LiH", ACCENT[2], "⁷Li/H")], "LiH", (5e-11, 2e-9), True),
    ]
    sweep = []
    for a, keys, obs, ylim, logy in specs:
        a.set_xscale("log")
        a.set_xlim(eta[0], eta[-1])
        a.set_ylim(*ylim)
        if logy:
            a.set_yscale("log")
        a.axvspan(eta_band[0] - eta_band[1], eta_band[0] + eta_band[1], color="#9ca3af", alpha=0.45, lw=0)
        v, e = OBSERVED[obs]
        a.axhspan(v - 2 * e, v + 2 * e, color=keys[0][1], alpha=0.15, lw=0)
        for k, c, lab in keys:
            sweep.append((a.plot([], [], lw=2, color=c, label=lab)[0], k))
        a.legend(fontsize=7, loc="upper right" if obs == "Yp" else "upper left", ncol=2)
        a.tick_params(labelsize=7)
        if a is not ax_l:
            a.set_xticklabels([])
    ax_y.set_title("Schramm plot: yields vs η (bands: observed ±2σ, grey: Planck)", fontsize=8)
    ax_l.set_xlabel("baryon-to-photon ratio η", fontsize=8)
    li_note = ax_l.text(0.03, 0.08, "", transform=ax_l.transAxes, fontsize=7, color="#b91c1c")

    k_grow = [round(j * (len(T) - 1) / (n_grow - 1)) for j in range(n_grow)]  # temperature steps to show

    def update(frame: int):
        k = k_grow[min(frame, n_grow - 1)]
        for i, line in enumerate(lines):
            line.set_data(T[: k + 1 : 2], np.maximum(X[: k + 1 : 2, i], 1e-30))  # every 2nd step is enough to draw
        clock.set_text(f"T = {T[k] * 1e3:.4g} keV\nt = {t[k]:.3g} s")
        m = 0 if frame < n_grow else round(len(eta) * min(1.0, (frame - n_grow + 1) / n_sweep))
        for line, key in sweep:
            line.set_data(eta[:m], curves[key][:m])
        if frame >= n_grow + n_sweep - 1:
            li_note.set_text(
                f"Planck η: Y_p = {mark['Yp']:.4f}, D/H = {mark['DH']:.2e}\n"
                f"⁷Li/H = {mark['LiH']:.1e} = {mark['LiH'] / OBSERVED['LiH'][0]:.1f}× observed: lithium problem"
            )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.1)
    return save_gif(fig, update, frame_indices(n_grow + n_sweep + 10), path)
