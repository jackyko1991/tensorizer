"""Animate the packets sloshing in the anharmonic well, with norm (CN vs explicit Euler) and energy panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    x: torch.Tensor,
    V: torch.Tensor,
    times: torch.Tensor,
    psi: torch.Tensor,
    x_mean: torch.Tensor,
    norms: torch.Tensor,
    energies: torch.Tensor,
    t_euler: torch.Tensor,
    n_euler: torch.Tensor,
    x0s: list[float],
    dt: float,
    path: Path,
) -> Path:
    xs, Vs, t = x.cpu().numpy(), V.cpu().numpy(), times.cpu().numpy()
    prob = (psi.abs() ** 2).cpu().numpy()  # (T, N, B)
    re0 = psi[:, :, 0].real.cpu().numpy()  # (T, N) real part of the first packet
    xm = x_mean.cpu().numpy()  # (T, B)
    nrm, E = norms.cpu().numpy(), energies.cpu().numpy()  # (T, B)
    te, ne = t_euler.cpu().numpy(), n_euler.cpu().numpy()  # (S,), (S, B)
    B = prob.shape[2]
    colors = ACCENT[:B]

    fig = plt.figure(figsize=(8.4, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax_n = fig.add_subplot(grid[0, 1])
    ax_e = fig.add_subplot(grid[1, 1])

    # left: V(x) in energy units; each packet drawn as |ψ|² standing on its own energy level ⟨H⟩
    x_view = x0s[0] + 2.5
    e_top = E[0].max() + 0.35 * (E[0].max() + 1)
    scale = 0.22 * e_top / prob.max()
    re_scale = 0.6 * scale * prob.max() / abs(re0).max()  # Re ψ drawn a bit smaller than |ψ|²
    ax.plot(xs, Vs, color="#6b7280", lw=2, label="V(x)")
    ax.set_xlim(-x_view, x_view)
    ax.set_ylim(0, e_top + 0.08 * e_top)
    ax.set_xlabel("x")
    ax.set_ylabel("energy   (packets: |ψ|² at height ⟨H⟩)")
    lines, dots = [], []
    for b in range(B):
        ax.axhline(E[0, b], color=colors[b], lw=0.6, alpha=0.4)
        (line,) = ax.plot([], [], color=colors[b], lw=2, label=f"|ψ|²,  x0 = {x0s[b]:.1f}")
        (dot,) = ax.plot([], [], "v", color=colors[b], ms=7)
        lines.append(line)
        dots.append(dot)
    (re_line,) = ax.plot([], [], color=colors[0], lw=0.8, ls="--", alpha=0.7, label=f"Re ψ,  x0 = {x0s[0]:.1f}")
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)

    # top right: norm against the step number (log–log) — CN stays at 1, explicit Euler explodes
    n_cn = round(t[-1] / dt)
    plain = FuncFormatter(lambda v, _: f"{v:.0e}" if v >= 1e4 else f"{v:g}")  # plain-text ticks (no mathtext)
    ax_n.set_xscale("log")
    ax_n.set_yscale("log")
    ax_n.set_xlim(1, n_cn)
    ax_n.set_ylim(0.3, max(10.0, 3 * ne.max()))
    for axis in (ax_n.xaxis, ax_n.yaxis):
        axis.set_major_locator(LogLocator(base=10 if axis is ax_n.xaxis else 1e4))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(plain)
    steps_e = te / dt  # (S,) step numbers of the Euler run
    n_axis = np.maximum(t / dt, 1)  # (T,) CN step numbers; t = 0 drawn at n = 1 on the log axis
    ax_n.plot(steps_e[1:], ne[1:, 0], "o-", ms=2, color="#dc2626", lw=1.5, label="explicit Euler")
    (norm_line,) = ax_n.plot([], [], color=colors[0], lw=2, label="Crank–Nicolson")
    ax_n.set_title("norm  Σ|ψ|²dx  vs step n", fontsize=9)
    ax_n.legend(loc="center right", fontsize=7)
    ax_n.tick_params(labelsize=7)

    # bottom right: energy of each packet, conserved by CN
    ax_e.set_xlim(0, t[-1])
    ax_e.set_ylim(0, 1.15 * E.max())
    ax_e.set_xlabel("t")
    ax_e.set_title("energy ⟨H⟩ (Crank–Nicolson)", fontsize=9)
    e_lines = [ax_e.plot([], [], color=colors[b], lw=2)[0] for b in range(B)]
    ax_e.tick_params(labelsize=7)

    def update(step: int):
        for b in range(B):
            lines[b].set_data(xs, E[0, b] + scale * prob[step, :, b])
            dots[b].set_data([xm[step, b]], [E[0, b] - 0.03 * e_top])
            e_lines[b].set_data(t[: step + 1], E[: step + 1, b])
        re_line.set_data(xs, E[0, 0] + re_scale * re0[step])
        norm_line.set_data(n_axis[: step + 1], nrm[: step + 1, 0])
        clock.set_text(f"t = {t[step]:5.2f}\n|norm−1| = {abs(nrm[step] - 1).max():.0e}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 72), path)
