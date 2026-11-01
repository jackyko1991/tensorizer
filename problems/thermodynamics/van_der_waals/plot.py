"""Animate the isotherm as T falls through T_c: the loop, its equal areas, the flat line and the growing dome."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(T: torch.Tensor, V: torch.Tensor, r: dict, critical: tuple, path: Path) -> Path:
    T_c, V_c, p_c = critical
    temps = T.numpy()
    v = V.numpy() / V_c  # (V,)  reduced volume V / V_c
    p = r["p"].numpy() / p_c  # (T, V)  reduced pressure p / p_c
    p_star = r["p_star"].numpy() / p_c  # (T,)  NaN above T_c
    v_l, v_g = r["V_l"].numpy() / V_c, r["V_g"].numpy() / V_c  # (T,)

    fig, (ax, ax_pT) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1.6, 1]})

    # Left: p–V plane with the current isotherm, its Maxwell line and the coexistence dome so far.
    for t in range(0, len(temps), max(1, len(temps) // 8)):
        ax.plot(v, p[t], color="#d1d5db", lw=0.8)  # faint reference isotherms
    curve = ax.plot([], [], color=ACCENT[0], lw=2)[0]
    flat = ax.plot([], [], color="black", lw=1.5)[0]
    dome = ax.plot([], [], color=ACCENT[3], lw=1.5, ls="--", label="coexistence dome")[0]
    ax.plot([1], [1], "*", color=ACCENT[1], ms=10, label="critical point")
    shading = []
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlim(v[0], 5.5)  # linear axis, so the two shaded areas really look equal
    ax.set_ylim(min(0.0, p[:, v < 1].min()) - 0.05, 1.6)  # the loop can dip below p = 0
    ax.set_xlabel("V / V_c")
    ax.set_ylabel("p / p_c")
    ax.legend(fontsize=8, loc="upper right")
    title = ax.set_title("")

    # Right: vapour-pressure curve p*(T), ending at the critical point.
    ax_pT.plot([1], [1], "*", color=ACCENT[1], ms=10)
    vapour = ax_pT.plot([], [], color=ACCENT[3], lw=2)[0]
    ax_pT.set_xlim(temps.min() / T_c, temps.max() / T_c)
    ax_pT.set_ylim(0, 1.15)
    ax_pT.set_xlabel("T / T_c")
    ax_pT.set_ylabel("p* / p_c")
    ax_pT.set_title("liquid–gas coexistence", fontsize=10)

    def update(step: int):
        for patch in shading:
            patch.remove()
        shading.clear()
        curve.set_data(v, p[step])
        done = ~np.isnan(p_star[: step + 1])  # temperatures below T_c reached so far
        if not np.isnan(p_star[step]):
            ps = p_star[step]
            inside = (v >= v_l[step]) & (v <= v_g[step])
            flat.set_data([v_l[step], v_g[step]], [ps, ps])
            shading.append(ax.fill_between(v, p[step], ps, where=inside & (p[step] < ps), color=ACCENT[0], alpha=0.3))
            shading.append(ax.fill_between(v, p[step], ps, where=inside & (p[step] > ps), color=ACCENT[1], alpha=0.3))
        else:
            flat.set_data([], [])
        # dome: liquid ends from low T up to the critical point, then gas ends back down
        dome_v = np.concatenate([v_l[: step + 1][done][::-1], [1.0], v_g[: step + 1][done]])
        dome_p = np.concatenate([p_star[: step + 1][done][::-1], [1.0], p_star[: step + 1][done]])
        dome.set_data(dome_v, dome_p)
        vapour.set_data(np.append(1.0, temps[: step + 1][done] / T_c), np.append(1.0, p_star[: step + 1][done]))
        title.set_text(f"T = {temps[step]:.0f} K = {temps[step] / T_c:.2f} T_c   (equal shaded areas)")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(temps)), path)
