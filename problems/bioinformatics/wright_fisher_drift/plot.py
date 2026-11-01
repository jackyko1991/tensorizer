"""Animate drift: trajectories, the spread of allele frequencies, heterozygosity decay, Hardy–Weinberg, fixation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from . import equation

FIXED, LOST, OPEN = "#2563eb", "#ea580c", "#9ca3af"
GENO = [("AA", "#2563eb"), ("Aa", "#16a34a"), ("aa", "#ea580c")]
N_LINES = 40  # trajectories drawn


def animate(
    p: torch.Tensor,
    G_obs: torch.Tensor,
    H_sim: torch.Tensor,
    H_theory: torch.Tensor,
    s_grid: torch.Tensor,
    u_sim: torch.Tensor,
    u_theory: torch.Tensor,
    N: int,
    s: float,
    n_fix: int,
    path: Path,
) -> Path:
    P = p.cpu().numpy()  # (R, G+1)
    geno = G_obs.cpu().numpy()  # (K, G+1, 3)
    H, H_th = H_sim.cpu().numpy(), H_theory.cpu().numpy()
    G = P.shape[1] - 1
    t = np.arange(G + 1)
    final = P[:N_LINES, -1]
    fate = np.where(final == 1, FIXED, np.where(final == 0, LOST, OPEN))

    fig = plt.figure(figsize=(9.6, 5.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    ax_p = fig.add_subplot(grid[0, :2])
    ax_hist = fig.add_subplot(grid[0, 2])
    ax_h = fig.add_subplot(grid[1, 0])
    ax_hw = fig.add_subplot(grid[1, 1])
    ax_u = fig.add_subplot(grid[1, 2])

    # --- allele-frequency trajectories, coloured by their fate at the end ---
    ax_p.set_xlim(0, G)
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.set_xlabel("generation $t$", labelpad=1)
    ax_p.set_ylabel("frequency $p$ of A")
    ax_p.set_title(f"Wright–Fisher drift, N = {N}, s = {s:g}: blue fixed, orange lost", fontsize=10)
    lines = [ax_p.plot([], [], lw=0.9, color=fate[r])[0] for r in range(min(N_LINES, len(P)))]
    clock = ax_p.text(0.01, 0.93, "", transform=ax_p.transAxes, fontsize=9)

    # --- distribution of p over all replicates ---
    edges = np.linspace(0, 1, 22)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_ylim(0, 1)
    ax_hist.set_xlabel("$p$", labelpad=1)
    ax_hist.set_ylabel("fraction of populations")
    ax_hist.set_title(f"all {len(P)} populations", fontsize=10)
    bars = ax_hist.bar(edges[:-1], np.zeros(21), width=edges[1] - edges[0], align="edge", color="#93c5fd", ec="#2563eb")
    fate_text = ax_hist.text(0.5, 0.9, "", transform=ax_hist.transAxes, ha="center", fontsize=8)

    # --- mean heterozygosity against the neutral prediction ---
    ax_h.plot(t, H_th / H[0], "--", color="#374151", lw=1.2, label=r"$(1 - 1/2N)^t$")
    h_line = ax_h.plot([], [], color="#dc2626", lw=1.6, label="simulation")[0]
    ax_h.set_xlim(0, G)
    ax_h.set_ylim(0, 1.05)
    ax_h.set_xlabel("generation $t$", labelpad=1)
    ax_h.set_title(r"heterozygosity $\langle 2pq\rangle / H_0$", fontsize=10)
    ax_h.legend(fontsize=7.5, loc="upper right")

    # --- Hardy–Weinberg: genotype frequencies of N individuals vs p², 2pq, q² ---
    x = np.linspace(0, 1, 101)
    curves = equation.hardy_weinberg(torch.from_numpy(x)).numpy()
    dots = []
    for g, (name, color) in enumerate(GENO):
        ax_hw.plot(x, curves[:, g], color=color, lw=1.2, label=name)
        dots.append(ax_hw.plot([], [], "o", ms=2.5, color=color)[0])
    ax_hw.set_xlim(0, 1)
    ax_hw.set_ylim(0, 1)
    ax_hw.set_xlabel("$p$", labelpad=1)
    ax_hw.set_title(r"Hardy–Weinberg $p^2,\ 2pq,\ q^2$", fontsize=10)
    ax_hw.legend(fontsize=7, loc="upper center", ncol=3, handlelength=1)

    # --- fixation probability of a single new mutant ---
    xs = (4 * N * s_grid).cpu().numpy()
    us, ut = u_sim.cpu().numpy(), u_theory.cpu().numpy()
    fine = torch.linspace(xs.min(), xs.max(), 200, dtype=torch.float64) / (4 * N)
    ax_u.plot(4 * N * fine.numpy(), equation.kimura_fixation(1 / (2 * N), N, fine).numpy(), color="#374151", lw=1.2)
    err = np.sqrt(np.maximum(us * (1 - us), 1e-12) / n_fix)
    ax_u.errorbar(xs, np.maximum(us, 1e-5), yerr=err, fmt="o", ms=4, color="#dc2626", capsize=2)
    ax_u.axhline(1 / (2 * N), color="#9ca3af", ls="--", lw=1)
    ax_u.text(xs.min(), 1.15 / (2 * N), "1/2N", fontsize=8, color="#6b7280")
    ax_u.set_yscale("log")
    lo, hi = max(ut.min() * 0.4, 1e-5), max(us.max(), ut.max()) * 2
    ax_u.set_ylim(lo, hi)
    ax_u.minorticks_off()  # fewer tick artists to redraw in every frame
    ticks = [v for v in (1e-4, 1e-3, 1e-2, 1e-1, 1.0) if lo <= v <= hi]
    ax_u.set_yticks(ticks, [f"{v:g}" for v in ticks])  # plain labels: no mathtext per frame
    ax_u.set_xlabel("$4Ns$", labelpad=1)
    ax_u.set_title("fixation of one new copy: Kimura", fontsize=10)

    def update(step: int):
        for r, line in enumerate(lines):
            line.set_data(t[: step + 1], P[r, : step + 1])
        clock.set_text(f"t = {step}")
        frac = np.histogram(P[:, step], bins=edges)[0] / len(P)
        for bar, f in zip(bars, frac):
            bar.set_height(f)
        fixed, lost = (P[:, step] == 1).mean(), (P[:, step] == 0).mean()
        fate_text.set_text(f"fixed {fixed:.0%}   lost {lost:.0%}")
        h_line.set_data(t[: step + 1], H[: step + 1] / H[0])
        for g in range(3):
            dots[g].set_data(P[: geno.shape[0], step], geno[:, step, g])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(G + 1, 40), path, fps=8)
