"""Animate both experiments: time reflection at a jump of ε, and growth inside the momentum gap, with the bands."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

BLUE, RED, GREY, AMBER, PURPLE = "#2563eb", "#dc2626", "#6b7280", "#f59e0b", "#9333ea"


def animate(
    x: torch.Tensor,
    times: torch.Tensor,
    D_jump: torch.Tensor,
    times_p: torch.Tensor,
    D_per: torch.Tensor,
    spectra: torch.Tensor,
    k: torch.Tensor,
    k_band: torch.Tensor,
    w_F: torch.Tensor,
    gap: tuple[float, float],
    fresnel: tuple[float, float],
    jump: tuple[float, float, float],
    modulation: tuple[float, float],
    path: Path,
) -> Path:
    xs, t = x.numpy(), times.numpy()
    Dj = D_jump.numpy()
    eps1, eps2, t_jump = jump
    alpha, omega = modulation
    lo, hi = gap
    k_gap = 0.5 * (lo + hi)

    fig = plt.figure(figsize=(8.4, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax_j, ax_s = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_eps, ax_b = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])

    # (a) temporal interface in real space
    view = np.abs(xs) < 85
    ax_j.set_xlim(-85, 85)
    ax_j.set_ylim(-1.8, 1.8)
    for level, color, name in ((fresnel[0], BLUE, "forward"), (fresnel[1], RED, "backward")):
        ax_j.axhline(abs(level), color=color, ls="--", lw=0.9, alpha=0.7)
        ax_j.text(84, abs(level) + 0.04, f"{name} {level:+.2f}", ha="right", va="bottom", fontsize=7, color=color)
    (line_j,) = ax_j.plot([], [], color="#111827", lw=1.0)
    ax_j.set_xlabel("x", fontsize=9)
    ax_j.set_ylabel("D(x, t)", fontsize=9)
    ax_j.set_title(f"sudden jump ε: {eps1:g} → {eps2:g} at t = {t_jump:g}", fontsize=10)
    label_j = ax_j.text(0.02, 0.04, "", transform=ax_j.transAxes, fontsize=8)
    ax_j.tick_params(labelsize=8)

    # (b) periodic modulation: spectrum |D_k(t)| on a log scale
    spec = spectra.numpy()
    ks = k.numpy()
    show = (ks > 0) & (ks < 2.4 * k_gap)
    order = np.argsort(ks[show])
    k_show = ks[show][order]
    spec_show = spec[:, show][:, order]  # (S, K_show)
    ax_s.axvspan(lo, hi, color="#fde68a", alpha=0.6, lw=0, label="gap (perturbative)")
    ax_s.plot(k_show, np.maximum(spec_show[0], 1e-12), color=GREY, lw=1, label="t = 0")
    (line_s,) = ax_s.plot([], [], color=PURPLE, lw=1.4, label="now")
    ax_s.set_yscale("log")
    ax_s.minorticks_off()
    ax_s.set_ylim(spec_show[0].max() * 1e-3, spec_show.max() * 5)
    ax_s.set_xlim(0, k_show[-1])
    ax_s.set_xlabel("wavenumber k", fontsize=9)
    ax_s.set_ylabel("|D_k(t)|", fontsize=9)
    ax_s.set_title(f"periodic ε = 1 + {alpha:g} cos {omega:g}t", fontsize=10)
    ax_s.legend(loc="upper right", fontsize=7)
    ax_s.tick_params(labelsize=8)

    # (c) the two permittivities ε(t)
    t_fine = np.linspace(0, max(t[-1], times_p[-1].item()), 1500)
    ax_eps.plot(t_fine, np.where(t_fine < t_jump, eps1, eps2), color=BLUE, lw=1.6, label="sudden jump")
    ax_eps.plot(t_fine, eps1 * (1 + alpha * np.cos(omega * t_fine)), color=PURPLE, lw=1.0, label="periodic")
    cursor = ax_eps.axvline(0, color=AMBER, lw=1.5)
    ax_eps.set_xlim(0, t_fine[-1])
    ax_eps.set_xlabel("time t", fontsize=9)
    ax_eps.set_ylabel("ε(t)", fontsize=9)
    ax_eps.legend(loc="center right", fontsize=7)
    ax_eps.tick_params(labelsize=8)

    # (d) Floquet bands: Re ω (folded into [0, Ω/2]) and the growth rate Im ω
    kb = k_band.numpy()
    re = np.abs(w_F.real.numpy())  # (K, 2)
    growth = w_F.imag.numpy().max(axis=-1)
    ax_b.axvspan(lo, hi, color="#fde68a", alpha=0.6, lw=0)
    ax_b.plot(np.repeat(kb, 2), re.reshape(-1), ".", color=BLUE, ms=1.5)
    ax_b.set_ylim(0, omega / 2 * 1.08)
    ax_b.set_xlim(0, kb[-1])
    ax_b.set_xlabel("wavenumber k", fontsize=9)
    ax_b.set_ylabel("Re ω (folded)", fontsize=9, color=BLUE)
    ax_b.set_title("Floquet bands from eig(M_k)", fontsize=10)
    ax_b.tick_params(labelsize=8)
    ax_g = ax_b.twinx()
    ax_g.plot(kb, growth, color=RED, lw=1.4)
    ax_g.set_ylim(0, max(growth.max(), 1e-3) * 1.15)
    ax_g.set_ylabel("growth Im ω", fontsize=9, color=RED)
    ax_g.tick_params(labelsize=8)
    ax_g.grid(False)

    t_p = times_p.numpy()

    def update(s: int):
        line_j.set_data(xs[view], Dj[s, view])
        phase = "before the jump" if t[s] < t_jump else "after: forward + time-reflected wave"
        label_j.set_text(f"t = {t[s]:.0f}   {phase}")
        line_s.set_data(k_show, np.maximum(spec_show[s], 1e-12))
        cursor.set_xdata([t_p[s], t_p[s]])

    fig.tight_layout()
    n_frames = min(len(t), len(t_p))  # the two runs share the time grid
    return save_gif(fig, update, frame_indices(n_frames, 70), path, fps=12)
