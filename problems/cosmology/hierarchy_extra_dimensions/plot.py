"""Animate gravity spreading into a compact dimension, the 1/r^(1+n) -> 1/r crossover and the alpha-lambda limits."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, save_gif

from . import equation, solve


def animate(r, ns, cr, zoom, add, m_star, krc, tev, krc_sel, dev, path: Path) -> Path:
    rr = r.cpu().numpy()
    ratio = cr["ratio"].cpu().numpy()
    Z = len(zoom)
    sweep = m_star[1:].cpu().numpy()
    S = len(sweep)
    R_all = add["R"][:, 1:].cpu().numpy()  # (6, S)
    alpha = add["alpha"][:, 0].cpu().numpy()
    exc = (add["torsion"][:, 1:] | add["lhc"][:, 1:]).cpu().numpy()
    exc[0] = True  # n = 1: solar-system size, excluded by planetary orbits
    lam_lee = add["lam_limits"]["Lee et al."]
    lam_kap = add["lam_limits"]["Kapner et al."]

    fig, axs = plt.subplots(2, 2, figsize=(9.0, 7.0))
    ax_f, ax_p, ax_e, ax_r = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # --- (b) crossover ---
    for i, n in enumerate(ns):
        ax_p.loglog(rr, ratio[i], color=ACCENT[i], lw=1.6, label=f"n = {n}")
        ax_p.loglog(rr, 1 / (equation.g4(n) * rr**n), color=ACCENT[i], lw=0.7, ls=":")  # pure 1/r^(1+n)
    ax_p.axhline(1, color="#111827", lw=0.8)
    ax_p.set_ylim(0.5, 1e6)
    ax_p.set_xlim(rr[0], rr[-1])
    ax_p.set_xlabel("distance $r/R$")
    ax_p.set_ylabel("$|\\phi|\\,r / G_4M$  (1 = Newton)")
    ax_p.legend(fontsize=7, ncol=2, loc="upper right")
    ax_p.set_title("$1/r^{1+n}$ inside (dotted) → $1/r$ outside", fontsize=9)
    marker = ax_p.axvline(1, color="#f59e0b", lw=1.5)

    # --- (c) alpha-lambda plane ---
    ax_e.fill_between([lam_lee, 1e15], 1, 1e3, color="#fecaca", alpha=0.7, lw=0)
    ax_e.text(2e-3, 30, "excluded: $\\alpha\\geq 1$, $\\lambda > 38.6\\,\\mu$m\n(Eöt-Wash, Lee et al. 2020)",
              fontsize=7,
              color="#991b1b")  # fmt: skip
    ax_e.plot([lam_lee], [1], "s", color="#991b1b", ms=5)
    ax_e.plot([lam_kap], [1], "D", color="#b45309", ms=4, label="Kapner 2007: 56 μm")
    ax_e.axvline(1.496e11, color="#9ca3af", ls="--", lw=0.8)
    ax_e.text(1.6e11, 0.15, "1 AU", fontsize=7, color="#6b7280")
    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlim(1e-19, 1e15)
    ax_e.set_ylim(0.1, 1e3)
    ax_e.set_xlabel("Yukawa range $\\lambda = R$ (m)")
    ax_e.set_ylabel("strength $\\alpha = 2n$")
    pts = ax_e.scatter(R_all[:, 0], alpha, c=["#dc2626"] * 6, s=40, zorder=3, edgecolors="#111827")
    labels = [ax_e.text(R_all[i, 0], alpha[i] * 1.25, f"n={n}", fontsize=7, ha="center") for i, n in enumerate(ns)]
    ax_e.legend(fontsize=7, loc="lower right")
    e_title = ax_e.set_title("", fontsize=9)

    # --- (d) Randall-Sundrum ---
    k, t = krc.cpu().numpy(), tev.cpu().numpy()
    ax_r.semilogy(k, t, color=ACCENT[3], lw=2)
    ax_r.axhspan(1e3, 1e4, color="#bbf7d0", alpha=0.7, lw=0)
    ax_r.text(9.2, 2.2e3, "1–10 TeV", fontsize=8, color="#166534")
    w_sel = math.exp(-math.pi * krc_sel)
    ax_r.plot([krc_sel], [2.435e18 * w_sel], "o", color="#111827")
    ax_r.set_xlabel("$k r_c$")
    ax_r.set_ylabel("$\\bar M_{Pl}\\,e^{-\\pi k r_c}$ (GeV)")
    ax_r.set_title(f"Randall–Sundrum: $kr_c$ = {krc_sel:g} → warp {w_sel:.1e}", fontsize=9)

    def update(f: int):
        w = float(zoom[min(f, Z - 1)])
        if f < Z:  # zoom phase: recompute the field slice in the new window
            ax_f.clear()
            sl = solve.slice_field(w, 56, dev)
            x = sl["x"].cpu().numpy()
            phi = sl["phi"].cpu().numpy()
            Fx, Fy = sl["Fx"].cpu().numpy(), sl["Fy"].cpu().numpy()
            ax_f.contourf(x, x, np.log10(-phi), levels=12, cmap="Blues")
            ax_f.streamplot(x, x, Fx, Fy, color="#1e3a8a", density=0.8, linewidth=0.6, arrowsize=0.6)
            for j in range(-int(w / (2 * math.pi)) - 1, int(w / (2 * math.pi)) + 2):
                yc = 2 * math.pi * j
                ax_f.axhline(yc + math.pi, color="#dc2626", ls="--", lw=0.8)
                if abs(yc) <= w:
                    ax_f.plot([0], [yc], "o", color="#f59e0b" if j else "#111827", ms=4)
            ax_f.set_xlim(-w, w)
            ax_f.set_ylim(-w, w)
            ax_f.set_aspect("equal")
            ax_f.set_xlabel("$x$ (our space, units of $R$)")
            ax_f.set_ylabel("$y$ (compact circle, unrolled)")
            ax_f.set_title(f"n = 1: field lines, window {2 * w:.2g} R\n"
                           + ("far: flat (spreads in 3D, $1/r$)" if w > 6 else "near: radial in 4D ($1/r^2$)" if w < 1.5
                              else "crossover at $r \\sim R$"), fontsize=9)  # fmt: skip
        marker.set_xdata([w])
        s = f - Z if f >= Z else 0
        pts.set_offsets(np.c_[R_all[:, s], alpha])
        pts.set_color(np.where(exc[:, s], "#dc2626", "#16a34a"))
        pts.set_edgecolor("#111827")
        for i in range(6):
            labels[i].set_position((R_all[i, s], alpha[i] * 1.25))
        e_title.set_text(f"ADD points, M* = {sweep[s]:.2f} TeV (red: excluded by\ntorsion balances, the LHC or orbits)")

    fig.tight_layout()
    return save_gif(fig, update, list(range(Z + S)), path, fps=8)
