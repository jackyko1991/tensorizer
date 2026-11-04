"""Animate the fits galaxy by galaxy, while the radial acceleration relation fills up point by point."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import equation
from .tensors import MS2_PER_KMS2_KPC

LABELS = {
    "baryons": "baryons only",
    "nfw": "baryons + NFW halo",
    "burkert": "baryons + Burkert core",
    "mond": "MOND (RAR)",
}
COLOURS = {"baryons": "#6b7280", "nfw": ACCENT[0], "burkert": ACCENT[2], "mond": ACCENT[1]}


def animate(data: dict, fit: dict, R_fine, curves: dict, path: Path, per_galaxy: int = 14, tail: int = 10) -> Path:
    names = data["names"]
    mask = data["mask"].cpu().numpy()
    R, V, err = (data[k].cpu().numpy() for k in ("R", "V", "err"))
    Rf = R_fine.cpu().numpy()
    cv = {m: c.cpu().numpy() for m, c in curves.items()}
    chi = {m: fit[m]["chi2_nu"].cpu().numpy() for m in curves}
    # radial acceleration relation with the MOND-fit mass-to-light ratios: g_obs = V²/R against g_bar = V²_bar/R
    V2_bar = equation.baryonic_v2(data["gas"], data["disk"], data["bul"], fit["mond"]["ups_d"], fit["mond"]["ups_b"])
    g_bar = (V2_bar / data["R"]).cpu().numpy() * MS2_PER_KMS2_KPC
    g_obs = (data["V"] ** 2 / data["R"]).cpu().numpy() * MS2_PER_KMS2_KPC
    a0 = float(fit["mond"]["a0"]) * MS2_PER_KMS2_KPC

    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    pts = ax.errorbar([], [], yerr=[], fmt="o", ms=3, color="#111827", ecolor="#9ca3af", lw=1, label="observed (SPARC)")
    lines = {m: ax.plot([], [], lw=2, color=COLOURS[m])[0] for m in curves}
    ax.set_xlabel("radius r (kpc)")
    ax.set_ylabel("rotation speed V (km/s)")

    gx = np.logspace(-12.3, -8.5, 200)
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlim(gx[0], gx[-1])
    ax_r.set_ylim(gx[0], gx[-1])
    ax_r.plot(gx, gx, "--", color="#9ca3af", lw=1, label="g_obs = g_bar (no dark matter)")
    rar = ax_r.plot([], [], lw=2, color=ACCENT[1], label="")[0]
    scat = [ax_r.plot([], [], "o", ms=2.5, alpha=0.7, color=ACCENT[g % len(ACCENT)])[0] for g in range(len(names))]
    ax_r.set_xlabel("baryonic (Newtonian) acceleration $g_{bar}$ (m/s²)")
    ax_r.set_ylabel("observed $g_{obs} = V^2/r$ (m/s²)")
    ax_r.set_title("radial acceleration relation", fontsize=10)
    ax_r.legend(fontsize=7, loc="upper left")

    def update(frame: int):
        g = min(frame // per_galaxy, len(names) - 1)
        k = frame - g * per_galaxy if frame < len(names) * per_galaxy else per_galaxy
        m_ = mask[g]
        nonlocal pts
        pts.remove()
        pts = ax.errorbar(R[g, m_], V[g, m_], yerr=err[g, m_], fmt="o", ms=3, color="#111827", ecolor="#9ca3af",
                          lw=1, label="observed (SPARC)")  # fmt: skip
        top = 1.25 * max(V[g, m_].max(), max(cv[m][g].max() for m in cv))
        ax.set_xlim(0, 1.05 * Rf[g, -1])
        ax.set_ylim(0, top)
        ax.set_title(f"{names[g]}  (D = {data['distance'][g]} Mpc)", fontsize=10)
        for j, m in enumerate(cv):  # curve j grows during frames 3 + 3j … 5 + 3j
            share = np.clip((k - 2 - 3 * j) / 3, 0, 1)
            n = int(round(share * Rf.shape[1]))
            lines[m].set_data(Rf[g, :n], cv[m][g, :n])
            lines[m].set_label(f"{LABELS[m]}  χ²/ν = {chi[m][g]:.1f}" if n else LABELS[m])
        ax.legend(fontsize=7, loc="upper left")
        for h in range(len(names)):
            show = h < g or (h == g and k >= 2)
            scat[h].set_data(g_bar[h, mask[h]] if show else [], g_obs[h, mask[h]] if show else [])
        if frame >= len(names) * per_galaxy:
            rar.set_data(gx, gx / (1 - np.exp(-np.sqrt(gx / a0))))
            rar.set_label(f"fit: $a_0$ = {a0:.2e} m/s²")
            ax_r.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(names) * per_galaxy + tail), path, fps=8)
