"""Animate sky after sky: the pulsar array, a few residual curves and the binned correlation converging to HD."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from . import solve
from .equation import hellings_downs


def animate(
    pulsars: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    zeta: torch.Tensor,
    rho: torch.Tensor,
    edges: torch.Tensor,
    binned: torch.Tensor,
    shown: list[int],
    path: Path,
) -> Path:
    k_total = r.shape[0]
    p = pulsars.numpy()
    lon, lat = np.arctan2(p[:, 1], p[:, 0]), np.arcsin(p[:, 2])
    centres = (edges[1:] + edges[:-1]) / 2
    spread = binned.std(dim=0)  # (B,) scatter of a bin between skies
    zz = np.linspace(0, 180, 361)
    hd = hellings_downs(torch.cos(torch.deg2rad(torch.tensor(zz)))).numpy()
    t_np, r_np, rho_np, zeta_np = t.numpy(), r.numpy(), rho.numpy(), zeta.numpy()
    some = np.random.default_rng(0).permutation(len(zeta_np))[:1500]  # pairs drawn as dots (keeps the GIF small)
    mid = len(t_np) // 2

    fig = plt.figure(figsize=(11.5, 3.9))
    ax_m = fig.add_subplot(1, 3, 1, projection="mollweide")
    ax_r = fig.add_subplot(1, 3, 2)
    ax_c = fig.add_subplot(1, 3, 3)

    lim = 1.5 * float(r_np[:, :, mid].std())
    sky = ax_m.scatter(lon, lat, c=r_np[0, :, mid], cmap="coolwarm", vmin=-lim, vmax=lim, s=24, edgecolors="#374151")
    for a, mark in zip(shown, ("A", "B", "C")):
        ax_m.text(lon[a], lat[a] + 0.12, mark, fontsize=8, ha="center", color="#111827")
    ax_m.set_xticklabels([])
    ax_m.tick_params(labelsize=7)
    ax_m.set_title(f"{len(p)} pulsars, colour = residual at t = {t_np[mid]:.1f} yr", fontsize=9)

    lines = [ax_r.plot([], [], lw=1.5, label=m)[0] for m in ("A", "B", "C")]
    ax_r.set_xlim(t_np[0], t_np[-1])
    lim_r = float(np.abs(r_np[:, shown]).max()) * 1.05
    ax_r.set_ylim(-lim_r, lim_r)
    ax_r.set_xlabel("time (yr)")
    ax_r.set_ylabel("timing residual (GW rms = 1)")
    ang = [np.degrees(np.arccos(np.clip(p[shown[0]] @ p[b], -1, 1))) for b in shown[1:]]
    ax_r.legend(loc="lower left", fontsize=7, title=f"ζ(A,B) = {ang[0]:.0f}°, ζ(A,C) = {ang[1]:.0f}°", title_fontsize=7)
    title_r = ax_r.set_title("sky realisation", fontsize=9)

    pairs = ax_c.scatter(zeta_np[some], rho_np[0, some], s=2, color="#d1d5db", zorder=1, label="pairs, this sky")
    ax_c.plot(zz, hd, color="#dc2626", lw=2, zorder=3, label="Hellings–Downs Γ(ζ)")
    (hd_fit,) = ax_c.plot([], [], color="#dc2626", lw=1, ls=":", zorder=3, label="A·Γ fit")
    (mono,) = ax_c.plot([], [], color="#f59e0b", lw=1.2, ls="--", zorder=2, label="monopole fit")
    (dip,) = ax_c.plot([], [], color="#16a34a", lw=1.2, ls="--", zorder=2, label="dipole fit")
    bars = [None]
    ax_c.axhline(0, color="#9ca3af", lw=1)
    ax_c.set_xlim(0, 180)
    ax_c.set_ylim(-0.7, 1.05)
    ax_c.set_xticks([0, 45, 90, 135, 180])
    ax_c.set_xlabel("angular separation ζ (deg)")
    ax_c.set_ylabel("correlation ρ")
    ax_c.legend(loc="upper center", fontsize=6.5, ncol=2, framealpha=0.95)
    title_c = ax_c.set_title("mean of skies", fontsize=9)

    def update(k: int):
        sky.set_array(r_np[k, :, mid])
        for line, a in zip(lines, shown):
            line.set_data(t_np, r_np[k, a])
        title_r.set_text(f"sky realisation {k + 1} of {k_total}")
        pairs.set_offsets(np.column_stack([zeta_np[some], rho_np[k, some]]))
        mean = binned[: k + 1].mean(0)
        err = spread / (k + 1) ** 0.5
        fits = solve.fit_models(centres, mean, err)
        if bars[0] is not None:
            bars[0].remove()
        bars[0] = ax_c.errorbar(
            centres.numpy(), mean.numpy(), yerr=err.numpy(), fmt="o", ms=4, color="#2563eb", capsize=2, zorder=4
        )
        c = np.cos(np.radians(zz))
        hd_fit.set_data(zz, fits["Hellings–Downs"][0] * hd)
        mono.set_data(zz, fits["monopole"][0] * np.ones_like(zz))
        dip.set_data(zz, fits["dipole"][0] * c)
        chi = {name: v[1] for name, v in fits.items()}
        title_c.set_text(
            f"mean of {k + 1} skies, χ²: HD {chi['Hellings–Downs']:.0f}, "
            f"mono {chi['monopole']:.0f}, dip {chi['dipole']:.0f}"
        )

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(k_total, k_total), path, fps=3)
