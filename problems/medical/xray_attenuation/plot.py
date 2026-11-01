"""Animate: rays sweep across the phantom and build the projection profile, then the energy is swept."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullLocator

from problems._common.plotting import ACCENT, frame_indices, save_gif

SWEEP, ENERGY_SWEEP = 40, 24  # logical steps of the two phases


def animate(
    x: torch.Tensor,
    depth: torch.Tensor,
    mu_maps: torch.Tensor,  # (E+1, Y, X): table energies, then the chosen energy last
    profiles: torch.Tensor,  # (E+1, Y, X) I(y)/I0 inside the body
    mono: torch.Tensor,  # (E+1, X) projection at each energy
    poly: torch.Tensor,  # (X,)
    e_all: torch.Tensor,  # (E+1,)
    contrast: torch.Tensor,  # (E, 3)
    hardening: tuple,
    path: Path,
) -> Path:
    x, depth = x.cpu().numpy(), depth.cpu().numpy()
    mu_maps, profiles = mu_maps.cpu().numpy(), profiles.cpu().numpy()
    mono, poly, e_all, contrast = mono.cpu().numpy(), poly.cpu().numpy(), e_all.cpu().numpy(), contrast.cpu().numpy()
    d_slab, i_slab, mu_eff, e_mean = (h.cpu().numpy() for h in hardening)
    n_e = len(e_all) - 1
    chosen = n_e  # index of the user-chosen energy

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    (ax_img, ax_prof), (ax_con, ax_bh) = axes
    extent = (x[0], x[-1], depth[-1], depth[0])
    img = ax_img.imshow(mu_maps[chosen], cmap=plt.get_cmap("bone", 24), vmin=0, vmax=0.8, extent=extent, aspect="auto")
    fig.colorbar(img, ax=ax_img, label=r"$\mu$ (1/cm)", fraction=0.05)
    ax_img.set_xlabel("x (cm)")
    ax_img.set_ylabel("depth along beam (cm)")
    ax_img.grid(False)
    rays = ax_img.scatter([], [], c=[], s=5, cmap="autumn", norm=LogNorm(1e-3, 1))  # colour = I/I0 along the ray
    title = ax_img.set_title("")

    ax_prof.set_xlim(x[0], x[-1])
    ax_prof.set_ylim(1e-4, 1.5)
    ax_prof.set_yscale("log")
    ax_prof.yaxis.set_minor_locator(NullLocator())  # fewer tick lines → faster frames
    ax_prof.set_xlabel("detector position x (cm)")
    ax_prof.set_ylabel(r"$I/I_0$")
    (prof_line,) = ax_prof.plot([], [], color=ACCENT[0], lw=1.8, label="monochromatic")
    (poly_line,) = ax_prof.plot([], [], color=ACCENT[1], lw=1.2, ls="--", label="tube spectrum")
    ax_prof.legend(loc="lower center", fontsize=8)
    ax_prof.set_title("projection image (one row)")

    names = ["lung", "bone", "iodine"]
    for k in range(3):
        ax_con.plot(e_all[:n_e], contrast[:, k], "o-", ms=3, color=ACCENT[k + 1], label=names[k])
    ax_con.axhline(0, color="#9ca3af", lw=0.8)
    e_mark = ax_con.axvline(e_all[chosen], color="#f59e0b", lw=1.5)
    ax_con.set_xlabel("photon energy (keV)")
    ax_con.set_ylabel("contrast vs tissue (1 cm)")
    ax_con.legend(fontsize=8, loc="upper right")

    ax_bh.plot(d_slab[1:], mu_eff, color=ACCENT[0], lw=1.8)
    ax_bh.set_xlabel("soft-tissue thickness (cm)")
    ax_bh.set_ylabel(r"$\mu_{\rm eff}$ (1/cm)", color=ACCENT[0])
    ax_bh.set_title("beam hardening")
    twin = ax_bh.twinx()
    twin.plot(d_slab, e_mean, color=ACCENT[4], lw=1.4, ls="--")
    twin.set_ylabel("mean energy (keV)", color=ACCENT[4])
    twin.grid(False)

    ny = len(depth)
    ray_rows = np.arange(0, ny, 2)

    def update(step: int):
        if step < SWEEP:  # phase 1: rays sweep across x at the chosen energy
            e_idx = chosen
            col = min(len(x) - 1, round(step * (len(x) - 1) / (SWEEP - 1)))
            prof_line.set_data(x[: col + 1], mono[e_idx, : col + 1])
            poly_line.set_data([], [])
            cols = [c for c in range(col - 8, col + 1, 4) if c >= 0]
        else:  # phase 2: sweep the photon energy over the table
            e_idx = min(n_e - 1, (step - SWEEP) * n_e // ENERGY_SWEEP)
            prof_line.set_data(x, mono[e_idx])
            poly_line.set_data(x, poly)
            cols = list(range(4, len(x), max(1, len(x) // 10)))
        img.set_data(mu_maps[e_idx])
        rx = np.repeat(x[cols], len(ray_rows))
        ry = np.tile(depth[ray_rows], len(cols))
        rays.set_offsets(np.column_stack([rx, ry]))
        rays.set_array(profiles[e_idx][np.ix_(ray_rows, cols)].T.ravel())
        e_mark.set_xdata([e_all[e_idx], e_all[e_idx]])
        title.set_text(f"phantom at E = {e_all[e_idx]:.0f} keV (beam ↓)")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(SWEEP + ENERGY_SWEEP, 64), path, fps=8)
