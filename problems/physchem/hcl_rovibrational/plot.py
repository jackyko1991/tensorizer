"""Animate the HCl band as the gas warms up: spectrum, an isotope doublet, rotational populations and the fit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif


def animate(grid, S, T, R, P, I_R, J, pop, fit, nu0, names, path: Path) -> Path:
    g, S, T = grid.cpu().numpy(), S.cpu().numpy(), T.cpu().numpy()
    R, P, J, pop, nu0 = R.cpu().numpy(), P.cpu().numpy(), J.cpu().numpy(), pop.cpu().numpy(), nu0.cpu().numpy()
    S = S / S.max(1, keepdims=True)  # each frame scaled to its strongest line, so the envelope shape is easy to read

    fig = plt.figure(figsize=(8.4, 5.0))
    grid_spec = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    ax_s = fig.add_subplot(grid_spec[0, :])
    ax_z = fig.add_subplot(grid_spec[1, 0])
    ax_f = fig.add_subplot(grid_spec[1, 1])
    ax_p = fig.add_subplot(grid_spec[1, 2])

    # --- the whole fundamental band ---
    ax_s.set_xlim(g[0], g[-1])
    ax_s.set_ylim(0, 1.25)
    ax_s.set_xlabel("wavenumber (cm⁻¹)", labelpad=1)
    ax_s.set_ylabel("absorption (arb.)")
    ax_s.axvline(nu0[0], color="#9ca3af", ls="--", lw=1)
    ax_s.text(nu0[0], 1.13, "ν₀: band gap, no Q branch", ha="center", fontsize=8, color="#6b7280")
    ax_s.text(0.5 * (nu0[0] + g[-1]), 1.05, "R branch  (ΔJ = +1)", ha="center", fontsize=9, color="#2563eb")
    ax_s.text(0.5 * (nu0[0] + g[0]), 1.05, "P branch  (ΔJ = −1)", ha="center", fontsize=9, color="#2563eb")
    for label, x in (("R(0)", R[0, 0]), ("P(1)", P[0, 1])):
        ax_s.text(
            x, -0.02, label, ha="center", va="top", fontsize=7, color="#dc2626", transform=ax_s.get_xaxis_transform()
        )
    (line_s,) = ax_s.plot([], [], color="#2563eb", lw=0.9)
    title = ax_s.set_title("HCl fundamental band")

    # --- zoom on one line: the H³⁵Cl / H³⁷Cl doublet, 3 : 1 by natural abundance ---
    j = min(3, len(J) - 2)
    centre = 0.5 * (R[0, j] + R[1, j])
    window = (g > centre - 5) & (g < centre + 5)
    ax_z.set_xlim(centre - 5, centre + 5)
    ax_z.set_ylim(0, S[:, window].max() * 1.25)
    ax_z.set_xlabel("cm⁻¹", labelpad=1)
    ax_z.set_title(f"R({j}) isotope doublet", fontsize=10)
    ax_z.set_yticks([])
    for i, name in enumerate(names):
        ax_z.axvline(R[i, j], color="#9ca3af", ls=":", lw=1)
        ax_z.text(R[i, j], 0.92, name, transform=ax_z.get_xaxis_transform(), ha="center", fontsize=7.5)
    (line_z,) = ax_z.plot([], [], color="#2563eb", lw=1.4)

    # --- combination-difference fit (static: line positions do not depend on T) ---
    x, y = fit["x"].cpu().numpy(), fit["y"][0].cpu().numpy()  # H³⁵Cl
    xs = np.linspace(0, x.max() * 1.05, 50)
    D = fit["D"][0].item()
    for s, (label, key, color) in enumerate((("v = 0", "B0", "#2563eb"), ("v = 1", "B1", "#dc2626"))):
        B = fit[key][0].item()
        ax_f.plot(x, y[s], "o", ms=3, color=color)
        ax_f.plot(xs, B - 2 * D * xs, color=color, lw=1, label=f"{label}: B = {B:.3f}")
    ax_f.set_xlabel("J² + J + 1", labelpad=1)
    ax_f.set_ylabel(r"$\Delta_2F\,/\,(4J+2)$  (cm⁻¹)", fontsize=8)
    ax_f.set_title("combination differences", fontsize=10)
    ax_f.legend(fontsize=7, loc="lower left")
    ax_f.text(
        0.97, 0.95, f"α_e = {fit['ae'][0].item():.3f}\nB_e = {fit['Be'][0].item():.3f}",
        transform=ax_f.transAxes, ha="right", va="top", fontsize=7.5,
    )  # fmt: skip

    # --- Boltzmann population of the rotational levels of v = 0 ---
    bars = ax_p.bar(J, pop[0, 0], color="#93c5fd", edgecolor="#1e40af", lw=0.6)
    ax_p.set_xlim(-0.7, J[-1] + 0.7)
    ax_p.set_ylim(0, pop[:, 0].max() * 1.08)
    ax_p.set_xlabel("J (v = 0)", labelpad=1)
    ax_p.set_title("level populations", fontsize=10)

    def update(step: int):
        line_s.set_data(g, S[step])
        line_z.set_data(g[window], S[step, window])
        for bar, h in zip(bars, pop[step, 0]):
            bar.set_height(h)
        title.set_text(f"HCl fundamental band  v = 0 → 1   ·   T = {T[step]:.0f} K")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(T), len(T)), path, fps=12)
