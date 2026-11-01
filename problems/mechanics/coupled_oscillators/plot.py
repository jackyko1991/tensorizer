"""Animate the chain of masses, with the strongest normal modes and the energy below it."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def zigzag(x_start: float, x_end: float, n_coils: int = 4, width: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Points of a spring drawn as a zigzag between x_start and x_end at height 0."""
    xs = np.linspace(x_start, x_end, 2 * n_coils + 3)
    ys = np.zeros_like(xs)
    ys[1:-1:2], ys[2:-1:2] = width, -width
    ys[-2] = 0.0
    return xs, ys


def animate(sol: dict, kinetic: torch.Tensor, potential: torch.Tensor, path: Path) -> Path:
    t = sol["times"].cpu().numpy()  # (T,)
    x = sol["x"].cpu().numpy()  # (T, N)
    e = sol["e"].cpu().numpy()  # (N, N) mode n = column n
    omega, A = sol["omega"].cpu().numpy(), sol["A"].cpu().numpy()  # (N,)
    ke, pe = kinetic.cpu().numpy(), potential.cpu().numpy()  # (T,)
    n = x.shape[1]
    rest = np.arange(1, n + 1)  # equilibrium positions, spacing 1 m; walls at 0 and N+1

    fig = plt.figure(figsize=(7, 4.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])
    ax_chain = fig.add_subplot(grid[0, :])
    ax_modes = fig.add_subplot(grid[1, 0])
    ax_energy = fig.add_subplot(grid[1, 1])

    # --- top: the chain itself ---
    ax_chain.set_xlim(-0.3, n + 1.3)
    ax_chain.set_ylim(-0.5, 0.5)
    ax_chain.set_yticks([])
    ax_chain.grid(False)
    ax_chain.set_xlabel("position (m)")
    ax_chain.axvspan(-0.25, 0, color="#9ca3af")  # left wall
    ax_chain.axvspan(n + 1, n + 1.25, color="#9ca3af")  # right wall
    springs = [ax_chain.plot([], [], color="#6b7280", lw=1)[0] for _ in range(n + 1)]
    masses = ax_chain.plot([], [], "s", ms=14, color=ACCENT[0])[0]
    clock = ax_chain.text(0.01, 0.92, "", transform=ax_chain.transAxes, va="top", fontsize=9)

    # --- bottom left: shapes of the three most excited modes e_{in} (static) ---
    strongest = np.argsort(A)[::-1][: min(3, n)]
    sites = np.arange(n + 2)
    for row, p in enumerate(strongest):
        shape = np.concatenate([[0.0], e[:, p] * np.sign(e[0, p] or 1.0), [0.0]])  # fixed ends
        offset = -1.2 * row
        ax_modes.axhline(offset, color="#d1d5db", lw=0.8)
        ax_modes.plot(sites, offset + shape, "o-", ms=3, color=ACCENT[(row + 1) % len(ACCENT)])
        ax_modes.text(n + 1.2, offset, f"n={p + 1}\nω={omega[p]:.2f}", fontsize=7, va="center")
    ax_modes.set_xlim(-0.3, n + 3.2)
    ax_modes.set_yticks([])
    ax_modes.set_xlabel("mass index i")
    ax_modes.set_title("strongest modes $e_{in}$", fontsize=9)

    # --- bottom right: energy vs time ---
    ax_energy.plot(t, ke, lw=1, label="kinetic")
    ax_energy.plot(t, pe, lw=1, label="potential")
    ax_energy.plot(t, ke + pe, lw=1.8, color="#111827", label="total")
    cursor = ax_energy.axvline(0, color="#9ca3af", lw=1)
    ax_energy.set_xlim(t[0], t[-1])
    ax_energy.set_ylim(0, (ke + pe).max() * 1.15)
    ax_energy.set_xlabel("t (s)")
    ax_energy.set_ylabel("energy (J)")
    ax_energy.legend(fontsize=7, loc="lower right")

    def update(step: int):
        pos = rest + x[step]  # (N,) where each mass is now
        ends = np.concatenate([[0.0], pos, [n + 1.0]])
        for s, spring in enumerate(springs):
            spring.set_data(*zigzag(ends[s], ends[s + 1]))
        masses.set_data(pos, np.zeros(n))
        cursor.set_xdata([t[step]])
        clock.set_text(f"t = {t[step]:.2f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), n_frames=100), path)
