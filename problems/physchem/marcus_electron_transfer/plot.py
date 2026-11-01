"""Animate: trajectories hopping between the two parabolas for each ΔG°, and the rate against driving force."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import crossing_point, product_surface, reactant_surface

REACT, PROD, CROSS = "#2563eb", "#dc2626", "#f59e0b"
LAM_COLOURS = ["#16a34a", "#2563eb", "#9333ea"]
N_SHOWN = 150  # trajectories drawn on the parabolas


def animate(
    dG: torch.Tensor,
    lams: torch.Tensor,
    temps: torch.Tensor,
    k: torch.Tensor,
    dG_sim: torch.Tensor,
    k_sim: torch.Tensor,
    k_err: torch.Tensor,
    res: dict,
    snaps: list[int],
    dt: float,
    lam: float,
    T: float,
    path: Path,
) -> Path:
    """k (L, T, G) on the grid dG (G,); simulation at dG_sim (D,); snapshots res['snap_x'] (S, D, N)."""
    dG, lams, temps, k = (a.cpu().numpy() for a in (dG, lams, temps, k))
    dG_sim, k_sim, k_err = (a.cpu().numpy() for a in (dG_sim, k_sim, k_err))
    sx = res["snap_x"][:, :, :N_SHOWN].cpu().numpy()
    ss = res["snap_state"][:, :, :N_SHOWN].cpu().numpy()
    n_snap, n_d = sx.shape[0], len(dG_sim)

    fig, (ax_g, ax_k) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1, 1.15]})
    xs = np.linspace(-0.9, 2.2, 300)
    (curve_r,) = ax_g.plot(xs, reactant_surface(xs, lam), color=REACT, lw=2, label="reactant  $G_R = \\lambda x^2$")
    (curve_p,) = ax_g.plot([], [], color=PROD, lw=2, label="product  $G_P = \\lambda(x-1)^2 + \\Delta G°$")
    (cross,) = ax_g.plot([], [], "o", color=CROSS, ms=8, zorder=4)
    dots = ax_g.scatter(np.zeros(N_SHOWN), np.zeros(N_SHOWN), s=10, zorder=3)
    ax_g.set_xlim(xs[0], xs[-1])
    ax_g.set_ylim(-2.2 * lam, 1.6 * lam)
    ax_g.set_xlabel("solvent coordinate x")
    ax_g.set_ylabel("free energy (eV)")
    ax_g.legend(fontsize=7, loc="lower left")
    title_g = ax_g.set_title("", fontsize=10)
    info = ax_g.text(0.98, 0.97, "", transform=ax_g.transAxes, ha="right", va="top", fontsize=8)

    floor = 1e-6 * k.max()
    for i, lm in enumerate(lams):
        ax_k.semilogy(
            -dG, np.maximum(k[i, 1], floor), color=LAM_COLOURS[i], lw=1.8, label=f"λ = {lm:.2f} eV, {T:.0f} K"
        )
    for j in (0, 2):
        ax_k.semilogy(
            -dG,
            np.maximum(k[1, j], floor),
            color=LAM_COLOURS[1],
            lw=1,
            ls=":",
            label=f"λ = {lam:.2f} eV, {temps[j]:.0f} K",
        )
    ax_k.axvline(lam, color="#9ca3af", lw=1, ls="--")
    ax_k.text(lam, floor * 3, " $-\\Delta G° = \\lambda$", fontsize=8, color="#6b7280")
    ax_k.text(0.97, 0.04, "inverted\nregion →", transform=ax_k.transAxes, ha="right", fontsize=8, color="#6b7280")
    pts = ax_k.errorbar([], [], yerr=[], fmt="o", color="#374151", ms=5, label="Langevin + Landau–Zener")
    marker = ax_k.axvline(0, color=CROSS, lw=1.5)
    ax_k.set_xlim(-dG[-1], -dG[0])
    ax_k.set_ylim(floor, 5 * k.max())
    ax_k.set_xlabel("driving force −ΔG° (eV)")
    ax_k.set_ylabel("rate k (1/ps)")
    ax_k.set_title("Marcus rate: rises, peaks at −ΔG° = λ, falls", fontsize=10)
    ax_k.legend(fontsize=7, loc="lower center")
    shown = {"bars": pts}

    def update(f: int):
        d, s = divmod(f, n_snap)
        g0 = dG_sim[d]
        curve_p.set_data(xs, product_surface(xs, lam, g0))
        xc = crossing_point(lam, g0)
        cross.set_data([xc], [reactant_surface(xc, lam)])
        x, st = sx[s, d], ss[s, d]
        y = np.where(st == 0, reactant_surface(x, lam), product_surface(x, lam, g0))
        dots.set_offsets(np.c_[x, y])
        dots.set_color(np.where(st[:, None] == 0, np.array([[0.15, 0.39, 0.92, 1]]), np.array([[0.86, 0.15, 0.15, 1]])))
        title_g.set_text(f"ΔG° = {g0:+.2f} eV,  t = {snaps[s] * dt:.2f} ps")
        info.set_text(f"on product: {st.mean():.0%}")
        shown["bars"].remove()
        shown["bars"] = ax_k.errorbar(
            -dG_sim[: d + 1], k_sim[: d + 1], yerr=k_err[: d + 1], fmt="o", color="#374151", ms=5, capsize=2
        )
        marker.set_xdata([-g0, -g0])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(n_d * n_snap, 54), path, fps=5)
