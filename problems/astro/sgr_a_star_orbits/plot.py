"""Animate S2 around Sgr A*: the ellipse on the sky with the data, the radial-velocity curve, the fit convergence."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import observables, period

N_FRAMES = 72


def animate(data: dict, fit: dict, path: Path) -> Path:
    k = fit["k"]
    p = fit["p"][k : k + 1]
    best = fit["best"]
    big_p = period(p).item()
    # times of one orbit, evenly spaced in eccentric anomaly so the fast pericentre passage gets frames too
    big_e = torch.linspace(-math.pi, math.pi, N_FRAMES, dtype=torch.float64)
    t_frames = best["tp"] + (big_e - best["e"] * torch.sin(big_e)) * big_p / (2 * math.pi)
    t_dense = torch.linspace(1990.0, 2020.0, 3000, dtype=torch.float64)
    ra_c, dec_c, rv_c = (x[0].numpy() for x in observables(p, t_dense))
    ra_f, dec_f, rv_f = (x[0].numpy() for x in observables(p, t_frames))
    tf, td = t_frames.numpy(), t_dense.numpy()

    fig, (ax_s, ax_v, ax_c) = plt.subplots(1, 3, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.05, 1.25, 0.8]})
    ax_s.errorbar(data["ra"].numpy(), data["dec"].numpy(), xerr=data["ra_err"].numpy(), yerr=data["dec_err"].numpy(),
                  fmt="o", ms=2.5, color="#93c5fd", ecolor="#bfdbfe", elinewidth=0.8, zorder=1,
                  label="S2 astrometry 1992–2016")  # fmt: skip
    ax_s.plot(ra_c, dec_c, color="#2563eb", lw=1.4, zorder=2, label="Kepler fit")
    ax_s.plot([0], [0], marker="+", ms=12, mew=2.5, color="#111827", zorder=4)
    ax_s.text(-6, 6, "Sgr A*", fontsize=8)
    (trail,) = ax_s.plot([], [], color="#dc2626", lw=2.2, zorder=3)
    (star,) = ax_s.plot([], [], "o", ms=8, color="#dc2626", mec="#7f1d1d", zorder=5)
    ax_s.set_xlim(ra_c.max() + 25, ra_c.min() - 25)  # east to the left, as on the sky
    ax_s.set_ylim(dec_c.min() - 20, dec_c.max() + 20)
    ax_s.set_aspect("equal")
    ax_s.set_xlabel("ΔRA (mas, east ←)")
    ax_s.set_ylabel("ΔDec (mas)")
    ax_s.legend(loc="center", fontsize=7)
    title_s = ax_s.set_title("", fontsize=9)

    ax_v.errorbar(data["t_v"].numpy(), data["rv"].numpy(), yerr=data["rv_err"].numpy(), fmt="o", ms=3,
                  color="#2563eb", ecolor="#93c5fd", label="radial velocity (VLT, Keck)")  # fmt: skip
    ax_v.plot(td, rv_c, color="#111827", lw=1.2, label="Kepler fit")
    ax_v.axvline(best["tp"], color="#9ca3af", ls="--", lw=1)
    ax_v.axvline(best["tp"] + big_p, color="#9ca3af", ls="--", lw=1)
    ax_v.text(
        best["tp"] + big_p - 0.3, rv_c.min(), "next pericentre\n(GRAVITY 2018)", fontsize=7, ha="right", color="#4b5563"
    )
    (marker,) = ax_v.plot([], [], "o", ms=8, color="#dc2626", mec="#7f1d1d", zorder=5)
    ax_v.set_xlim(1999, 2019.5)
    ax_v.set_ylim(rv_c.min() - 300, rv_c.max() + 300)
    ax_v.set_xlabel("year")
    ax_v.set_ylabel("line-of-sight velocity (km/s)")
    ax_v.legend(loc="upper center", fontsize=7)
    ax_v.set_title(
        f"M = {best['M']:.2f} ± {fit['errors']['M']:.2f} × 10⁶ M☉,  "
        f"R0 = {best['R0']:.2f} ± {fit['errors']['R0']:.2f} kpc,"
        f"  P = {big_p:.2f} yr",
        fontsize=9,
    )

    hist = fit["history"].numpy()  # (iterations + 1, B)
    n_res = data["n_res"]
    for b in range(hist.shape[1]):
        ax_c.plot(hist[:, b] / n_res, color="#d1d5db", lw=0.7)
    ax_c.plot(hist[:, k] / n_res, color="#dc2626", lw=2, label="best start")
    ax_c.axhline(1.0, color="#16a34a", ls="--", lw=1, label="χ²/N = 1")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Levenberg–Marquardt step")
    ax_c.set_ylabel("χ² / N")
    ax_c.legend(loc="upper right", fontsize=7)
    ax_c.set_title(f"{hist.shape[1]} starts fitted in one batch", fontsize=9)
    for ax in (ax_s, ax_v, ax_c):
        ax.tick_params(labelsize=7)

    def update(i: int):
        lo = max(0, i - 12)
        trail.set_data(ra_f[lo : i + 1], dec_f[lo : i + 1])
        star.set_data([ra_f[i]], [dec_f[i]])
        t_now = tf[i] if tf[i] > 1999 else tf[i] + big_p  # show the RV marker inside the data window
        rv_now = np.interp(t_now, td, rv_c)
        marker.set_data([t_now], [rv_now])
        title_s.set_text(f"S2 around Sgr A*, year {tf[i]:.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(N_FRAMES, N_FRAMES), path, fps=12)
