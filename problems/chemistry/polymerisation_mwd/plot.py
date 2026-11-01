"""Animate polymerisation: the step-growth and living distributions on a common x / X_n axis, Đ and Carothers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif

BLUE, RED, ORANGE, GREY, DARK = "#2563eb", "#dc2626", "#ea580c", "#9ca3af", "#374151"
U_MAX = 4.0  # right edge of the x / X_n axis


def animate(x, p_k, s_k, P_k, poisson, kmc, p_kmc, xn, disp, xn_mc, disp_mc, path: Path) -> Path:
    xs = x.cpu().numpy()  # (L,)
    p, s = p_k.cpu().numpy(), s_k.cpu().numpy()  # (K,)
    P, mc, po = P_k.cpu().numpy(), kmc.cpu().numpy(), poisson.cpu().numpy()  # (K, 2, L), (K, L), (K, L)
    pm = p_kmc.cpu().numpy()
    Xn, D = xn.cpu().numpy(), disp.cpu().numpy()  # (K, 2)
    Xn_mc, D_mc = xn_mc.cpu().numpy(), disp_mc.cpu().numpy()  # (K,)
    xn_end = Xn[-1, 0]

    def density(Pk: np.ndarray, xn_k: float) -> np.ndarray:
        """X_n w(x): the weight distribution as a density in u = x / X_n (area 1)."""
        w = xs * Pk / max((xs * Pk).sum(), 1e-300)
        return xn_k * w

    fig = plt.figure(figsize=(8.4, 4.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])
    ax_s = fig.add_subplot(grid[0, 0])
    ax_l = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    # --- step growth: ODE, KMC, Flory ---
    (ode_s,) = ax_s.plot([], [], color=BLUE, lw=1.8, label="population balance")
    (mc_s,) = ax_s.plot([], [], "o", color=ORANGE, ms=2.5, label="kinetic Monte Carlo")
    (fl_s,) = ax_s.plot([], [], "--", color=DARK, lw=1, label="Flory $x(1-p)^2p^{x-1}$")
    ax_s.set_xlim(0, U_MAX)
    ax_s.set_ylim(0, 0.6)
    ax_s.set_title("step growth (A–B monomers)", fontsize=9)
    ax_s.set_ylabel("$X_n\\,w(x)$", fontsize=8)
    ax_s.tick_params(labelsize=7)
    ax_s.legend(fontsize=6.5, loc="upper right")
    text_s = ax_s.text(0.02, 0.88, "", transform=ax_s.transAxes, fontsize=8)

    # --- living chain growth: ODE, Poisson ---
    (ode_l,) = ax_l.plot([], [], color=RED, lw=1.8, label="population balance")
    (po_l,) = ax_l.plot([], [], "--", color=DARK, lw=1, label="Poisson")
    (fl_l,) = ax_l.plot([], [], color=BLUE, lw=1, alpha=0.35, label="step growth, same $X_n$")
    ax_l.set_xlim(0, U_MAX)
    ax_l.set_title("living chain growth (no termination)", fontsize=9)
    ax_l.set_xlabel("chain length relative to the mean, $x / X_n$", fontsize=8)
    ax_l.set_ylabel("$X_n\\,w(x)$", fontsize=8)
    ax_l.tick_params(labelsize=7)
    ax_l.legend(fontsize=6.5, loc="upper right")
    text_l = ax_l.text(0.02, 0.88, "", transform=ax_l.transAxes, fontsize=8)

    # --- dispersity against X_n ---
    xg = np.linspace(1, xn_end, 300)
    nu = xg - 1
    ax_d.plot(xg, 2 - 1 / xg, "--", color=BLUE, lw=1, alpha=0.6)
    ax_d.plot(xg, 1 + nu / (1 + nu) ** 2, "--", color=RED, lw=1, alpha=0.6)
    ax_d.plot(xg, 1 + 1 / xg, ":", color=RED, lw=1, alpha=0.6)
    ax_d.text(xn_end, 2.03, "Đ = 1 + p → 2", ha="right", va="bottom", fontsize=7, color=BLUE)
    ax_d.text(xn_end, 1.04, "Đ ≈ 1 + 1/$X_n$", ha="right", va="bottom", fontsize=7, color=RED)
    (d_step,) = ax_d.plot([], [], color=BLUE, lw=1.8)
    (d_liv,) = ax_d.plot([], [], color=RED, lw=1.8)
    (d_mc,) = ax_d.plot([], [], "o", color=ORANGE, ms=2.5)
    ax_d.set_xscale("log")
    ax_d.minorticks_off()
    ax_d.set_xlim(1, xn_end * 1.05)
    ax_d.set_ylim(0.95, 2.15)
    ax_d.set_title("dispersity Đ = $X_w / X_n$", fontsize=9)
    ax_d.tick_params(labelsize=7)

    # --- Carothers ---
    pg = np.linspace(0, p[-1], 300)
    ax_c.plot(pg, 1 / (1 - pg), color=GREY, lw=2.5, alpha=0.6, label="Carothers $1/(1-p)$")
    (c_ode,) = ax_c.plot([], [], "o", color=BLUE, ms=5, label="population balance")
    (c_mc,) = ax_c.plot([], [], "x", color=ORANGE, ms=6, label="KMC")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, xn_end * 1.1)
    ax_c.set_xlabel("conversion p", fontsize=8)
    ax_c.set_ylabel("$X_n$", fontsize=8)
    ax_c.tick_params(labelsize=7)
    ax_c.legend(fontsize=6.5, loc="upper left")

    bins = np.unique(np.round(np.geomspace(1, xs[-1] + 1, 70)).astype(int))  # lumps for the noisy KMC tail

    def update(k: int):
        xn_k = Xn[k, 0]
        u = xs / xn_k
        ode_s.set_data(u, density(P[k, 0], xn_k))
        fl_s.set_data(u, xn_k * xs * (1 - p[k]) ** 2 * p[k] ** (xs - 1))
        w_mc = xs * mc[k] / max((xs * mc[k]).sum(), 1e-300)  # KMC weight fraction, then averaged over bins
        centres, dens = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            if hi > lo:
                centres.append(0.5 * (lo + hi - 1))
                dens.append(w_mc[lo - 1 : hi - 1].sum() / (hi - lo))
        mc_s.set_data(np.array(centres) / Xn_mc[k], np.array(dens) * Xn_mc[k])
        text_s.set_text(f"p = {p[k]:.3f}   $X_n$ = {xn_k:.1f}   Đ = {D[k, 0]:.3f}")

        dl = density(P[k, 1], Xn[k, 1])
        ode_l.set_data(xs / Xn[k, 1], dl)
        po_l.set_data(xs / Xn[k, 1], density(po[k], Xn[k, 1]))
        fl_l.set_data(u, density(P[k, 0], xn_k))
        ax_l.set_ylim(0, max(0.6, 1.15 * dl.max()))
        text_l.set_text(f"ν = {s[k]:.1f}   $X_n$ = {Xn[k, 1]:.1f}   Đ = {D[k, 1]:.3f}")

        d_step.set_data(Xn[: k + 1, 0], D[: k + 1, 0])
        d_liv.set_data(Xn[: k + 1, 1], D[: k + 1, 1])
        d_mc.set_data(Xn_mc[: k + 1], D_mc[: k + 1])
        c_ode.set_data([p[k]], [xn_k])
        c_mc.set_data([pm[k]], [Xn_mc[k]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(p), 90), path, fps=10)
