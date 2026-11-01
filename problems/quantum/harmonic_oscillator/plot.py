"""Animate |ψ|² swinging in the parabola, with the lowest eigenstates drawn faintly at their energies."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    x: torch.Tensor,
    times: torch.Tensor,
    psi: torch.Tensor,
    phi: torch.Tensor,
    energies: torch.Tensor,
    x_mean: torch.Tensor,
    omega: float,
    x0: float,
    path: Path,
    n_show: int = 6,
) -> Path:
    xs = x.cpu().numpy()
    prob = (psi.abs() ** 2).cpu().numpy()  # (T, N)
    t = times.cpu().numpy()
    xm = x_mean.cpu().numpy()  # (T,)
    E = energies[:n_show].cpu().numpy()
    states = phi[:, :n_show].T.cpu().numpy()  # (n_show, N)

    x_view = abs(x0) + 4 / math.sqrt(omega)
    e_top = max(E[-1] + omega, 0.5 * omega**2 * x0**2 * 1.2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(-x_view, x_view)
    ax.set_ylim(0, e_top)
    ax.set_xlabel("x")
    ax.set_ylabel("energy  (V(x), E_n)")
    ax.plot(xs, 0.5 * omega**2 * xs**2, color="#6b7280", lw=2, label="V = ½ω²x²")
    scale = 0.35 * omega / abs(states).max()  # eigenstate wiggles fit between the rungs
    for En, state in zip(E, states):  # faint stationary states φ_n(x) drawn at height E_n
        ax.axhline(En, color="#e5e7eb", lw=0.8)
        ax.plot(xs, En + scale * state, color="#9ca3af", lw=0.8, ls="--")

    # |ψ|² on a second axis so its height is independent of the energy scale
    ax_p = ax.twinx()
    ax_p.grid(False)
    ax_p.set_ylim(0, 1.6 * prob.max())
    ax_p.set_ylabel("|ψ|²", color=ACCENT[0])
    (line,) = ax_p.plot([], [], color=ACCENT[0], lw=2, label="|ψ|²")
    fill = [ax_p.fill_between(xs, 0 * xs, color=ACCENT[0], alpha=0.2)]
    (mean_dot,) = ax_p.plot([], [], "v", color=ACCENT[0], ms=9, label="mean position <x>")
    (ball,) = ax_p.plot([], [], "o", color=ACCENT[1], ms=8, label="classical x0 cos ωt")
    clock = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", family="monospace")
    ax_p.legend(loc="upper right", fontsize=8)

    def update(step: int):
        line.set_data(xs, prob[step])
        fill[0].remove()
        fill[0] = ax_p.fill_between(xs, prob[step], color=ACCENT[0], alpha=0.2)
        mean_dot.set_data([xm[step]], [1.35 * prob.max()])
        ball.set_data([x0 * math.cos(omega * t[step])], [1.45 * prob.max()])
        clock.set_text(f"t = {t[step]:5.2f}\n<x> = {xm[step]:+.3f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path)
