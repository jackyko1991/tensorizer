"""Animate Hodgkin–Huxley: voltage traces for four stimulus currents, the gates m, h, n, and the f–I curve."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .solve import T_ON

SHOWN = [0.1, 0.25, 0.5, 1.0]  # traces drawn for these fractions of I_max


def animate(
    times: torch.Tensor,
    traj: torch.Tensor,
    I_ext: torch.Tensor,
    rate: torch.Tensor,
    n_spikes: list[int],
    path: Path,
) -> Path:
    stride = max(1, len(times) // 1500)
    t = times[::stride].cpu().numpy()
    y = traj[::stride].cpu().numpy()  # (T, 4, B)
    cur, f = I_ext.cpu().numpy(), rate.cpu().numpy()
    shown = [int(np.abs(cur - s * cur[-1]).argmin()) for s in SHOWN]
    colors = [f"C{k}" for k in range(len(shown))]

    fig = plt.figure(figsize=(8.4, 4.8))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    grid = outer[0].subgridspec(len(shown), 1, hspace=0.12)
    right = outer[1].subgridspec(2, 1, hspace=0.55)

    # --- membrane potential of four axons ---
    traces = []
    for k, b in enumerate(shown):
        ax = fig.add_subplot(grid[k])
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-85, 90)  # headroom above the spikes for the label
        ax.set_yticks([-65, 0, 40])
        ax.axvspan(T_ON, t[-1], color="#fef3c7", lw=0)  # the current is on
        ax.text(
            0.99,
            0.96,
            f"I = {cur[b]:.1f} µA/cm²",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=colors[k],
        )
        traces.append(ax.plot([], [], lw=1.3, color=colors[k])[0])
        if k < len(shown) - 1:
            ax.tick_params(labelbottom=False)
        if k == 0:
            ax.set_title("membrane potential V (mV); shaded: current on", fontsize=10)
    ax.set_xlabel("time (ms)")

    # --- f–I curve: steady firing rate against current ---
    ax_f = fig.add_subplot(right[0])
    single = np.array([n > 0 and r == 0 for n, r in zip(n_spikes, f)])
    rep = f > 0
    ax_f.plot(cur[rep], f[rep], "o-", ms=3, lw=1, color="#374151", label="repetitive firing")
    ax_f.plot(cur[single], np.zeros(single.sum()), "x", ms=5, color="#dc2626", label="single spike only")
    ax_f.plot(
        cur[~rep & ~single], np.zeros((~rep & ~single).sum()), "o", ms=3, mfc="white", color="#9ca3af", label="no spike"
    )
    for k, b in enumerate(shown):
        ax_f.plot([cur[b]], [f[b]], "o", ms=8, mfc="none", mew=1.5, color=colors[k])
    ax_f.set_xlim(-0.03 * cur[-1], cur[-1] * 1.03)
    ax_f.set_ylim(-5, max(f.max(), 1) * 1.15)
    ax_f.set_xlabel("stimulus I (µA/cm²)", labelpad=1)
    ax_f.set_ylabel("rate (Hz)")
    ax_f.set_title("f–I curve", fontsize=10)
    ax_f.legend(loc="upper left", fontsize=7)

    # --- gating variables of the strongest shown stimulus ---
    ax_g = fig.add_subplot(right[1])
    ax_g.set_xlim(T_ON - 5, min(t[-1], T_ON + 45))
    ax_g.set_ylim(0, 1)
    ax_g.set_xlabel("time (ms)", labelpad=1)
    ax_g.set_title(f"gates at I = {cur[shown[-1]]:.1f} µA/cm²", fontsize=10)
    gate_colors = ["#2563eb", "#f59e0b", "#16a34a"]
    gates = [ax_g.plot([], [], lw=1.5, color=c)[0] for c in gate_colors]
    for j, name in enumerate("mhn"):  # label each gate at its resting value, left of the step
        ax_g.text(T_ON - 4, y[0, 1 + j, shown[-1]] + 0.04, name, color=gate_colors[j], fontsize=10, style="italic")

    def update(step: int):
        for k, b in enumerate(shown):
            traces[k].set_data(t[: step + 1], y[: step + 1, 0, b])
        for j in range(3):
            gates[j].set_data(t[: step + 1], y[: step + 1, 1 + j, shown[-1]])

    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.1)
    return save_gif(fig, update, frame_indices(len(t), 90), path)
