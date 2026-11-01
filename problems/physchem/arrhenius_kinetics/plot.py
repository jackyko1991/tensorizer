"""Animate the runs: [A](t) at every temperature, the three straight-line tests, and the Arrhenius plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from problems._common.plotting import frame_indices, save_gif

from .equation import R_GAS

N_FRAMES = 50  # frames on a log-spaced common clock
LABELS = ("[A] (zeroth order)", "ln[A] (first order)", "1/[A] (second order)")


def fmt_time(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.3g} s"
    if seconds < 7200:
        return f"{seconds / 60:.3g} min"
    if seconds < 3 * 86400:
        return f"{seconds / 3600:.3g} h"
    return f"{seconds / 86400:.3g} d"


def animate(
    tau: torch.Tensor,
    conc: torch.Tensor,
    t_end: torch.Tensor,
    T: torch.Tensor,
    tau_s: torch.Tensor,
    data: torch.Tensor,
    beta: torch.Tensor,
    R2: torch.Tensor,
    best: int,
    k_true: torch.Tensor,
    k_fit: torch.Tensor,
    Ea_fit: float,
    A_fit: float,
    path: Path,
) -> Path:
    """conc (N+1, T): smooth curves of the measured order; data (T, S): noisy samples at τ_s (S,)."""
    tau, conc, t_end, T = (x.cpu().numpy() for x in (tau, conc, t_end, T))
    tau_s, data, beta, R2 = (x.cpu().numpy() for x in (tau_s, data, beta, R2))
    k_true, k_fit = k_true.cpu().numpy(), k_fit.cpu().numpy()
    n_T = len(T)
    colors = plt.get_cmap("plasma")(np.linspace(0.0, 0.85, n_T))  # dark = cold, orange = hot
    g_data = np.stack([data, np.log(np.clip(data, 1e-6, None)), 1 / np.clip(data, 1e-6, None)])  # (3, T, S)

    fig = plt.figure(figsize=(9.0, 4.4))
    grid = fig.add_gridspec(3, 3, width_ratios=[1.15, 0.9, 1.0])
    ax_c = fig.add_subplot(grid[:, 0])
    ax_g = [fig.add_subplot(grid[i, 1]) for i in range(3)]
    ax_a = fig.add_subplot(grid[:, 2])

    # --- concentration curves against real time (log axis): hot runs finish first ---
    t_real = tau[:, None] * t_end[None, :]  # (N+1, T) s
    t_samp = tau_s[None, :] * t_end[:, None]  # (T, S) s
    clock = np.geomspace(0.02 * t_end.min(), t_end.max(), N_FRAMES)  # one common clock for the animation
    ax_c.set_xscale("log")
    ax_c.minorticks_off()  # fewer tick artists to redraw in every frame
    ax_c.set_xlim(clock[0], clock[-1] * 1.05)
    ax_c.set_ylim(0, 1.08)
    ax_c.set_xlabel("time $t$ (s, log scale)")
    ax_c.set_ylabel("[A] (M)")
    ax_c.set_title("measured runs: warmer = faster", fontsize=10)
    curves = [ax_c.plot([], [], lw=1.6, color=colors[t], label=f"{T[t]:.0f} K")[0] for t in range(n_T)]
    dots_c = [ax_c.plot([], [], "o", ms=3, color=colors[t])[0] for t in range(n_T)]
    ax_c.legend(fontsize=7, loc="lower left")
    time_text = ax_c.text(0.97, 0.97, "", transform=ax_c.transAxes, ha="right", va="top", fontsize=9)

    # --- the three candidate straight lines ---
    dots_g = []
    for g, ax in enumerate(ax_g):
        ok = "  ✓" if g == best else ""
        ax.set_title(f"{LABELS[g]}  R² = {R2[g].mean():.4f}{ok}", fontsize=8, pad=2,
                     color="#15803d" if g == best else "#374151")  # fmt: skip
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(3))  # few ticks: fewer artists to redraw per frame
        lo, hi = np.nanmin(g_data[g]), np.nanmax(g_data[g])
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        for t in range(n_T):  # fitted line, drawn back in reduced time: y = β0 + β1 t_end τ
            ax.plot(tau, beta[g, t, 0] + beta[g, t, 1] * t_end[t] * tau, lw=0.8, color=colors[t], alpha=0.6)
        dots_g.append([ax.plot([], [], "o", ms=2.2, color=colors[t])[0] for t in range(n_T)])
        if g < 2:
            ax.set_xticklabels([])
    ax_g[2].set_xlabel(r"$t\,/\,t_{\mathrm{end}}$  (each run to 90 % conversion)", fontsize=8)

    # --- Arrhenius plot: ln k against 1000/T ---
    x = 1000.0 / T
    x_line = np.linspace(x.min() - 0.03, x.max() + 0.03, 50)
    y_fit = np.log(np.clip(k_fit, 1e-300, None))
    (line,) = ax_a.plot([], [], color="#374151", lw=1.4, label="LSQ fit")
    ax_a.plot(x, np.log(k_true), "x", color="#9ca3af", ms=6, label="true $k$")
    points = ax_a.scatter(x, y_fit, c=colors, s=28, zorder=3, label="fitted $k$")
    ax_a.set_xlim(x_line[0], x_line[-1])
    ax_a.set_ylim(min(y_fit.min(), np.log(k_true).min()) - 0.4, max(y_fit.max(), np.log(k_true).max()) + 0.4)
    ax_a.set_xlabel("1000 / T  (1/K)")
    ax_a.set_ylabel(r"$\ln k$")
    ax_a.set_title("Arrhenius plot", fontsize=10)
    fit_text = ax_a.text(0.97, 0.97, f"slope = $-E_a/R$\n$E_a$ = {Ea_fit / 1e3:.1f} kJ/mol\n$A$ = {A_fit:.2e}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=8)  # fmt: skip
    ax_a.legend(fontsize=7, loc="lower left")

    def update(step: int):
        now = clock[step]
        for t in range(n_T):
            i = int(np.searchsorted(t_real[:, t], now, side="right"))  # grid points of run t done by now
            n_shown = int(np.searchsorted(t_samp[t], now, side="right"))  # samples measured so far
            curves[t].set_data(t_real[1:i, t], conc[1:i, t])  # skip t = 0, which a log axis cannot show
            dots_c[t].set_data(t_samp[t, 1:n_shown], data[t, 1:n_shown])
            for g in range(3):
                dots_g[g][t].set_data(tau_s[:n_shown], g_data[g, t, :n_shown])
        done = now >= t_end * (1 - 1e-9)  # runs that have finished: their k goes on the Arrhenius plot
        points.set_offsets(np.c_[x, np.where(done, y_fit, np.nan)])
        all_done = bool(done.all())  # the straight-line fit needs every temperature
        line.set_data(x_line, np.log(A_fit) - Ea_fit / R_GAS * x_line / 1000.0)
        line.set_visible(all_done)
        fit_text.set_visible(all_done)
        time_text.set_text(f"t = {fmt_time(now)}")

    fig.tight_layout(pad=0.6, w_pad=0.8, h_pad=0.4)
    return save_gif(fig, update, frame_indices(N_FRAMES), path)
