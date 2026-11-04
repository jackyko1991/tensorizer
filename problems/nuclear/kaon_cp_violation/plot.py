"""Top: what a K⁰ beam is made of as it flies. Middle: the T-violating asymmetry plateau. Bottom: ππ interference."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from . import tensors

BLUE, RED, GREY, AMBER = "#2563eb", "#dc2626", "#9ca3af", "#f59e0b"


def animate(t: torch.Tensor, out: dict, eps_delta: tuple, path: Path) -> Path:
    eps, delta = eps_delta
    t = t.cpu().numpy()
    p = out["p"].cpu().numpy()  # (B, T, final, initial)
    kaon, anti = p[0, :, 0, 0], p[0, :, 1, 0]  # a beam that started as K⁰
    total = kaon + anti
    frac = kaon / total
    a_t = 1e3 * out["a_t"].cpu().numpy()  # (B, T) in 10⁻³
    a_cpt = 1e3 * out["a_cpt"].cpu().numpy()
    pipi = out["pipi"].cpu().numpy()  # (B, T, start)
    a_pipi = (pipi[..., 1] - pipi[..., 0]) / (pipi[..., 1] + pipi[..., 0])  # (B, T)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylabel("beam content")
    ax1.set_title(r"a pure K$^0$ beam: strangeness oscillates at $\Delta m$ while K$_S$ dies out", fontsize=10)
    ax1b = ax1.twinx()
    ax1b.set_yscale("log")
    ax1b.set_ylim(max(total.min(), 1e-12) / 2, 1.5)
    ax1b.set_ylabel("kaons left", color=GREY)
    ax1b.grid(False)
    ax1.plot([], [], color=BLUE, lw=6, label=r"K$^0$")
    ax1.plot([], [], color=RED, lw=6, alpha=0.6, label=r"$\bar{\mathrm{K}}^0$")
    ax1.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    (survive,) = ax1b.plot([], [], color="#374151", lw=1.2)

    value, error = tensors.CPLEAR_AT
    ax2.axhspan(1e3 * (value - error), 1e3 * (value + error), color=AMBER, alpha=0.25, lw=0, label="CPLEAR 1998")
    ax2.axhline(4e3 * eps.real, color=AMBER, lw=1, ls="--", label=r"$4\,\mathrm{Re}\,\varepsilon$")
    (line_t,) = ax2.plot([], [], color=BLUE, lw=2, label=r"$A_T$ (Kabir)")
    (line_cpt,) = ax2.plot([], [], color="#9333ea", lw=1.5, label=r"$A_{CPT}$")
    (line_zero,) = ax2.plot([], [], color=GREY, lw=1.5, label=r"$\varepsilon=\delta=0$")
    lo = min(-1.0, a_cpt[0].min() - 1)
    hi = max(1e3 * (value + error), a_t[0, 1:].max(), a_cpt[0].max()) + 2
    ax2.set_ylim(lo, hi)
    ax2.set_ylabel(r"asymmetry ($10^{-3}$)")
    ax2.legend(loc="center right", fontsize=7, ncol=3, framealpha=0.9)

    ax3.axhline(0, color=GREY, lw=0.8)
    (line_pp,) = ax3.plot([], [], color=BLUE, lw=2, label=r"$(\bar R-R)/(\bar R+R)$ to $\pi^+\pi^-$")
    (line_pp0,) = ax3.plot([], [], color=GREY, lw=1.5, label=r"$\varepsilon = 0$")
    span = np.abs(a_pipi[0]).max()
    ax3.set_ylim(-1.15 * span, 1.15 * span)
    ax3.set_ylabel("ππ asymmetry")
    ax3.set_xlabel(r"proper time $t$ ($\tau_S$ = 0.0895 ns)")
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.9)
    cursors = [ax.axvline(t[0], color="#111827", lw=1) for ax in (ax1, ax2, ax3)]
    readout = ax2.text(0.01, 0.06, "", transform=ax2.transAxes, fontsize=8)
    fills = []

    def update(step: int):
        for f in fills:
            f.remove()
        fills.clear()
        s = slice(0, step + 1)
        fills.append(ax1.fill_between(t[s], 0, frac[s], color=BLUE, lw=0))
        fills.append(ax1.fill_between(t[s], frac[s], 1, color=RED, alpha=0.6, lw=0))
        survive.set_data(t[s], total[s])
        line_t.set_data(t[1 : step + 1], a_t[0, 1 : step + 1])
        line_cpt.set_data(t[s], a_cpt[0, s])
        line_zero.set_data(t[1 : step + 1], a_t[2, 1 : step + 1])
        line_pp.set_data(t[s], a_pipi[0, s])
        line_pp0.set_data(t[s], a_pipi[2, s])
        for c in cursors:
            c.set_xdata([t[step]])
        if step > 0:
            readout.set_text(f"$A_T$ = {a_t[0, step]:.2f}e-3   $A_{{CPT}}$ = {a_cpt[0, step]:.2f}e-3")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path)
