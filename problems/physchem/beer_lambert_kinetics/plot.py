"""Animate the run: the pure spectra, the measured spectra changing in time, and the recovered concentrations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

COLORS = ["#2563eb", "#16a34a", "#dc2626"]  # A, B, C
N_HISTORY = 12  # faint earlier spectra kept on screen


def animate(
    lam: torch.Tensor,
    E: torch.Tensor,
    t: torch.Tensor,
    c: torch.Tensor,
    c_hat: torch.Tensor,
    D: torch.Tensor,
    iso_ab: list[float],
    iso_bc: list[float],
    path: Path,
) -> Path:
    lam, E, t = lam.cpu().numpy(), E.cpu().numpy(), t.cpu().numpy()
    c, c_hat, D = c.cpu().numpy() * 1e6, c_hat.cpu().numpy() * 1e6, D.cpu().numpy()  # µM
    n_t = len(t)

    fig = plt.figure(figsize=(8.6, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[0.8, 1.2])
    ax_e = fig.add_subplot(grid[0, 0])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[:, 1])

    # --- pure-component spectra (columns of E) and their crossings ---
    for s, name in enumerate("ABC"):
        ax_e.plot(lam, E[:, s] / 1e3, color=COLORS[s], lw=1.6, label=name)
    for ax in (ax_e, ax_d):
        for x in iso_ab:
            ax.axvline(x, color="#f59e0b", ls="--", lw=1)
        for x in iso_bc:
            ax.axvline(x, color="#9333ea", ls=":", lw=1.2)
    ax_e.set_xlim(lam[0], lam[-1])
    ax_e.set_ylim(0, E.max() / 1e3 * 1.35)  # headroom for the legend
    ax_e.set_ylabel(r"$\varepsilon$ ($10^3$ L/(mol cm))", fontsize=8)
    title = r"columns of $E$: pure spectra   (-- $\varepsilon_A=\varepsilon_B$,  ⋯ $\varepsilon_B=\varepsilon_C$)"
    ax_e.set_title(title, fontsize=8.5, pad=2)
    ax_e.legend(fontsize=7, ncol=3, loc="upper right")
    ax_e.tick_params(labelsize=7)
    ax_e.set_xticklabels([])

    # --- measured spectra D(λ, t): faint history + the current one ---
    ax_d.set_xlim(lam[0], lam[-1])
    ax_d.set_ylim(min(0.0, D.min()) - 0.02, D.max() * 1.1)
    ax_d.set_xlabel("wavelength λ (nm)")
    ax_d.set_ylabel("absorbance D")
    ax_d.tick_params(labelsize=8)
    history_steps = np.linspace(0, n_t - 1, N_HISTORY).round().astype(int)
    shade = plt.get_cmap("viridis")(np.linspace(0, 0.9, N_HISTORY))
    history = [ax_d.plot([], [], lw=0.8, color=shade[i], alpha=0.6)[0] for i in range(N_HISTORY)]
    (current,) = ax_d.plot([], [], lw=1.8, color="#111827")
    clock = ax_d.text(0.98, 0.95, "", transform=ax_d.transAxes, ha="right", va="top", fontsize=9)

    # --- concentrations: true (lines) vs recovered from the spectra (dots) ---
    ax_c.set_xlim(0, t[-1])
    ax_c.set_ylim(min(0.0, c_hat.min()) - 1, c[0].sum() * 1.12)
    ax_c.set_xlabel("time t (s)")
    ax_c.set_ylabel("concentration (µM)")
    ax_c.set_title(r"recovered $\hat c = E^{+} D / l$ (dots) vs true (lines)", fontsize=9)
    for s, name in enumerate("ABC"):
        ax_c.plot(t, c[:, s], color=COLORS[s], lw=1.4, alpha=0.8, label=f"[{name}]")
    ax_c.plot(t, c.sum(1), color="#374151", ls="--", lw=1, label="[A]+[B]+[C]")
    dots = [ax_c.plot([], [], "o", ms=2.5, color=COLORS[s])[0] for s in range(3)]
    (dots_sum,) = ax_c.plot([], [], ".", ms=2.5, color="#374151")
    cursor = ax_c.axvline(0, color="#9ca3af", lw=1)
    ax_c.legend(fontsize=7, loc="center right")

    def update(step: int):
        for i, h in enumerate(history_steps):
            if h <= step:
                history[i].set_data(lam, D[h])
            else:
                history[i].set_data([], [])
        current.set_data(lam, D[step])
        clock.set_text(f"t = {t[step]:.0f} s")
        for s in range(3):
            dots[s].set_data(t[: step + 1], c_hat[: step + 1, s])
        dots_sum.set_data(t[: step + 1], c_hat[: step + 1].sum(1))
        cursor.set_xdata([t[step], t[step]])

    fig.tight_layout(pad=0.6, h_pad=0.4, w_pad=1.0)
    return save_gif(fig, update, frame_indices(n_t, 61), path)
