"""Animate the horizons growing as VSL / inflation are switched on, next to the conformal (light-cone) diagram."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import save_gif

from .tensors import A_REC

GREY, BLUE, GREEN = "#6b7280", "#2563eb", "#16a34a"


def animate(x, std, vsl, infl, stats, c_ratio, efolds, a_star, path: Path) -> Path:
    every = max(1, len(x) // 1200)  # thin the curves for drawing
    la = (x[::every] / math.log(10)).cpu().numpy()  # log10 a
    lg = lambda t: np.log10(np.maximum(t[:, ::every].cpu().numpy(), 1e-300))  # noqa: E731  (χ_p starts at 0)
    rH_s, chi_s = lg(std["hubble_radius"]), lg(std["chi"])
    rH_v, chi_v = lg(vsl["hubble_radius"]), lg(vsl["chi"])
    rH_i, chi_i = lg(infl["hubble_radius"]), lg(infl["chi"])
    s_std, s_vsl, s_inf = ({k: v.cpu().numpy() for k, v in s.items()} for s in stats)
    D = s_std["distance"][0]
    r, N = c_ratio.cpu().numpy(), efolds.cpu().numpy()

    fig, (ax, ax_c) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})
    ax.axhline(np.log10(D), color="#f59e0b", lw=1.2, ls="--")
    ax.text(-24, np.log10(D) - 3, "distance to the CMB surface D", color="#b45309", fontsize=7)
    ax.axvline(np.log10(A_REC), color="#9ca3af", lw=1, ls=":")
    ax.text(np.log10(A_REC) - 1, -33, "recombination", rotation=90, fontsize=7, color=GREY, ha="right")
    ax.axvline(np.log10(a_star), color="#9ca3af", lw=1, ls=":")
    ax.text(np.log10(a_star) + 1, -33, "$a_*$", fontsize=8, color=GREY)
    ax.plot(la, rH_s[0], color=GREY, lw=1.2, label="ΛCDM  c/(aH)")
    ax.plot(la, chi_s[0], color=GREY, lw=1.2, ls="--", label="ΛCDM  horizon $\\chi_p$")
    l_rv = ax.plot([], [], color=BLUE, lw=1.6, label="VSL  c(a)/(aH)")[0]
    l_cv = ax.plot([], [], color=BLUE, lw=1.6, ls="--", label="VSL  horizon")[0]
    l_ri = ax.plot([], [], color=GREEN, lw=1.3, label="inflation  c/(aH)")[0]
    l_ci = ax.plot([], [], color=GREEN, lw=1.3, ls="--", label="inflation  horizon")[0]
    ax.set_xlim(la[0], 0)
    ax.set_ylim(-36, 14)
    ax.set_xlabel("$\\log_{10} a$")
    ax.set_ylabel("$\\log_{10}$ comoving length (Mpc)")
    ax.legend(fontsize=6.5, loc="lower right", ncol=1)
    title = ax.set_title("placeholder", fontsize=9)

    # conformal diagram: light moves at 45°; lengths in units of D, recombination at η = 0
    ax_c.set_xlim(-1.7, 1.7)
    ax_c.set_ylim(-2.3, 1.15)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("comoving distance / D")
    ax_c.set_ylabel("conformal time $(\\eta - \\eta_{rec})$ / D")
    ax_c.plot([-1, 0, 1], [0, 1, 0], color="#f59e0b", lw=1.5)  # our past light cone back to last scattering
    ax_c.plot([0], [1], "*", ms=11, color="#111827")
    ax_c.text(0.08, 1.0, "us", fontsize=8)
    ax_c.plot([-1, 1], [0, 0], "o", color="#dc2626", ms=6)
    ax_c.axhline(0, color="#dc2626", lw=0.8, ls=":")
    ax_c.text(1.05, 0.05, "CMB", color="#dc2626", fontsize=8)
    ax_c.grid(False)
    fills = []
    bang_lines = {
        key: ax_c.plot([], [], color=col, lw=2)[0] for key, col in (("std", GREY), ("vsl", BLUE), ("inf", GREEN))
    }
    c_title = ax_c.set_title("placeholder\nplaceholder", fontsize=8)
    bottom = fig.text(0.5, 0.01, "", ha="center", fontsize=8.5)

    def cone_depth(s, f):
        return min(s["chi_rec"][f] / D, 2.25)

    def update(f: int):
        l_rv.set_data(la, rH_v[f])
        l_cv.set_data(la, chi_v[f])
        l_ri.set_data(la, rH_i[f])
        l_ci.set_data(la, chi_i[f])
        title.set_text(f"VSL: $c_{{early}}/c_0 = 10^{{{np.log10(r[f]):.1f}}}$     inflation: N = {N[f]:.1f} e-folds")
        for p in fills:
            p.remove()
        fills.clear()
        for key, s, col in (("vsl", s_vsl, BLUE), ("inf", s_inf, GREEN), ("std", s_std, GREY)):
            k = 0 if key == "std" else f
            d = cone_depth(s, k)
            for side in (-1, 1):  # past light cones of the two antipodal CMB points
                fills.append(ax_c.fill([side - d, side, side + d], [-d, 0, -d], color=col, alpha=0.18, lw=0)[0])
            bang_lines[key].set_data([-1.7, 1.7], [-d, -d] if d < 2.25 else [np.nan, np.nan])
        solved = [name for name, s in (("VSL", s_vsl), ("inflation", s_inf)) if s["solved"][f]]
        verdict = "overlap for " + " and ".join(solved) if solved else "do not overlap: horizon problem"
        c_title.set_text("past light cones of opposite CMB points\n" + verdict)
        bottom.set_text(f"causally disconnected patches on the CMB sky:  ΛCDM {s_std['patches'][0]:,.0f}   "
                        f"VSL {s_vsl['patches'][f]:,.0f}   inflation {s_inf['patches'][f]:,.0f}")  # fmt: skip

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_gif(fig, update, list(range(len(r))) + [len(r) - 1] * 12, path, fps=12)  # hold the end
