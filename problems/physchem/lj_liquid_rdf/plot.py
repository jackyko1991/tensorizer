"""Animate the production run: a thin slice of each box, g(r) building up, and the conserved energy."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .tensors import NAMES

COLOURS = ["#16a34a", "#2563eb", "#9333ea"]
SLAB = 1.0  # σ: thickness of the slice drawn from each box


def animate(res: dict, L: torch.Tensor, rho: torch.Tensor, T: torch.Tensor, dt: float, path: Path) -> Path:
    """res['frames'] (F, S, N, 3); res['g_running'] (F, S, K) on res['r'] (K,); res['E'] (K_steps, S)."""
    frames = res["frames"].cpu().numpy()
    g_run, r = res["g_running"].cpu().numpy(), res["r"].cpu().numpy()
    E = res["E"].cpu().numpy()
    E_rel = (E - E[0]) * 1e4  # in units of 10⁻⁴ ε
    L, rho, T = L.cpu().numpy(), rho.cpu().numpy(), T.cpu().numpy()
    steps_f = res["frame_step"]
    t = np.arange(1, len(E) + 1) * dt
    S = frames.shape[1]

    fig = plt.figure(figsize=(9, 5.2))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    boxes = [fig.add_subplot(grid[0, s]) for s in range(S)]
    ax_g = fig.add_subplot(grid[1, :2])
    ax_e = fig.add_subplot(grid[1, 2])

    dots = []
    for s, ax in enumerate(boxes):
        dots.append(
            ax.scatter([], [], s=max(4.0, 900 / L[s] ** 2), color=COLOURS[s], edgecolors="#1f2937", linewidths=0.3)
        )
        ax.set_xlim(0, L[s])
        ax.set_ylim(0, L[s])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{NAMES[s]}: ρ* = {rho[s]:.2f}, T* = {T[s]:.2f}", fontsize=9)

    ax_g.axhline(1, color="#9ca3af", lw=1, ls="--")
    ax_g.axvline(2 ** (1 / 6), color="#f59e0b", lw=1, ls=":")
    ax_g.text(2 ** (1 / 6), 0.2, " $2^{1/6}\\sigma$", color="#b45309", fontsize=8)
    lines = [ax_g.plot([], [], color=COLOURS[s], lw=1.6, label=NAMES[s])[0] for s in range(S)]
    ax_g.set_xlim(0, r[-1])
    ax_g.set_ylim(0, 1.1 * max(np.nanmax(g_run[-1]), 3.0))
    ax_g.set_xlabel("pair distance r (σ)")
    ax_g.set_ylabel("g(r)")
    ax_g.legend(fontsize=8)
    title_g = ax_g.set_title("", fontsize=10)

    spread = max(np.abs(E_rel).max(), 1.0)
    ax_e.set_xlim(0, t[-1])
    ax_e.set_ylim(-1.5 * spread, 1.5 * spread)
    elines = [ax_e.plot([], [], color=COLOURS[s], lw=1.2)[0] for s in range(S)]
    ax_e.set_xlabel("time (τ)")
    ax_e.set_ylabel("E/N − E/N at start (10⁻⁴ ε)")
    ax_e.set_title("NVE: energy is conserved", fontsize=10)

    def update(f: int):
        for s in range(S):
            x = frames[f, s]
            slab = np.abs(x[:, 2] - 0.5 * L[s]) < 0.5 * SLAB
            dots[s].set_offsets(x[slab, :2])
            lines[s].set_data(r, g_run[f, s])
        k = steps_f[f] + 1
        for s in range(S):
            elines[s].set_data(t[:k], E_rel[:k, s])
        title_g.set_text(f"radial distribution function, averaged up to t = {k * dt:.1f} τ")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(steps_f), 40), path, fps=6)
