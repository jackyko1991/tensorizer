"""Quadrupole and octopole maps with their axes, and the Monte Carlo test filling up frame by frame."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import equation, tensors


def multipole_map(a: torch.Tensor, ell: int, n_lat: int = 60, n_lon: int = 120):
    """T_ℓ(lon, lat) = Σ_m a_{ℓm} Y_{ℓm} on a lon/lat grid, lon from +180° to −180° (Galactic maps grow to the left)."""
    lon = torch.linspace(180, -180, n_lon, dtype=torch.float64)
    lat = torch.linspace(-90, 90, n_lat, dtype=torch.float64)
    n = tensors.unit_vectors(lon[None, :].expand(n_lat, -1), lat[:, None].expand(-1, n_lon))
    Y = equation.spherical_harmonics(n.reshape(-1, 3), ell)[:, ell * ell :]  # only this ℓ
    T = (Y @ a).real.reshape(n_lat, n_lon)
    return np.radians(-lon.numpy()), np.radians(lat.numpy()), T.numpy()


def _marks(ax, n, style, color, label, **kw):
    for sign in (1, -1):  # an axis has two ends
        lon, lat = tensors.lonlat(sign * n)
        lon = (lon + 180) % 360 - 180
        ax.plot(math.radians(-lon), math.radians(lat), style, color=color, label=label if sign == 1 else None, **kw)


def animate(alm, result, cos_sims, path: Path) -> Path:
    fig = plt.figure(figsize=(9.2, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    for row, ell in enumerate((2, 3)):
        ax = fig.add_subplot(grid[row, 0], projection="mollweide")
        x, y, T = multipole_map(alm[ell], ell)
        lim = np.abs(T).max()
        ax.pcolormesh(x, y, T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto", rasterized=True)
        _marks(ax, result["axis2"], "o", ACCENT[2], "quadrupole axis", ms=12, mfc="none", mew=2)
        _marks(ax, result["axis3"], "s", ACCENT[4], "octopole axis", ms=5, mec="white", mew=0.6)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color="#9ca3af", lw=0.4)
        ax.set_title(f"ℓ = {ell} (WMAP 9-yr ILC), ±{lim:.0f} μK", fontsize=8)
        if row == 1:
            ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)

    ax_p = fig.add_subplot(grid[:, 1])
    sims = np.sort(1 - cos_sims.numpy())  # (K,) 1 − |cos| of the isotropic skies
    K = len(sims)
    x_obs = 1 - result["cos"]
    xs = np.geomspace(1e-5, 1, 60)
    ax_p.loglog(xs, xs, color="#6b7280", lw=1, ls="--", label="isotropy: P = 1 − |cos| (uniform)")
    mc = ax_p.loglog([], [], "o", ms=3.5, color=ACCENT[0], label="random Gaussian skies")[0]
    ax_p.axvline(x_obs, color=ACCENT[1], lw=1.5, label=f"WMAP: angle {result['angle']:.1f}°")
    ax_p.set_xlim(1e-5, 1)
    ax_p.set_ylim(1e-5, 1.5)
    ax_p.set_xlabel(r"$1 - |\hat n_2\cdot\hat n_3|$  (0 = perfectly aligned)")
    ax_p.set_ylabel("share of skies at least this aligned")
    ax_p.legend(fontsize=7, loc="upper left")
    title = ax_p.set_title("", fontsize=8.5)
    probes = np.geomspace(1e-5, 1, 26)
    sizes = [max(200, int(K * (i + 1) / 60)) for i in range(60)]

    def update(frame: int):
        k = min(sizes[frame], K)
        s = np.sort(sims[:k]) if k < K else sims
        share = np.searchsorted(s, probes, side="right") / k  # empirical P(1 − |cos| ≤ x)
        keep = share > 0
        mc.set_data(probes[keep], share[keep])
        p = np.mean(s <= x_obs)
        title.set_text(f"{k:,} isotropic skies: p = {p:.4f}  (one-sided, no look-elsewhere)")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(60, 60), path)
