"""Animate the cups as thermometer bars next to their cooling curves T(t)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    times: torch.Tensor, temps: torch.Tensor, exact: torch.Tensor, k: torch.Tensor, T_env: float, path: Path
) -> Path:
    t = times.cpu().numpy()
    T = temps.cpu().numpy()  # (T, B)
    T_exact = exact.cpu().numpy()  # (T, B)
    n_cups = T.shape[1]
    colors = [ACCENT[b % len(ACCENT)] for b in range(n_cups)]
    cmap = plt.get_cmap("coolwarm")
    T_low, T_high = T_env - 5, 100.0

    fig, (ax_bar, ax_T) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1, 1.6]})

    # Left: one "thermometer" bar per cup, coloured from blue (cold) to red (hot).
    cups = list(range(1, n_cups + 1))
    bars = ax_bar.bar(cups, T[0] - T_low, bottom=T_low, edgecolor=colors, linewidth=2)
    ax_bar.axhline(T_env, color="gray", ls="--", lw=1)
    ax_bar.set_ylim(T_low, T_high)
    ax_bar.set_xticks(cups)
    ax_bar.set_xlabel("cup")
    ax_bar.set_ylabel("T (°C)")
    clock = ax_bar.set_title("")

    # Right: the simulated curves grow over time; dashed lines are the exact exponentials.
    lines = []
    for b in range(n_cups):
        ax_T.plot(t, T_exact[:, b], color=colors[b], ls="--", lw=1, alpha=0.5)
        lines.append(ax_T.plot([], [], color=colors[b], lw=2, label=f"k = {k[b]:.3f}")[0])
    ax_T.axhline(T_env, color="gray", ls="--", lw=1, label="room")
    ax_T.set_xlim(0, t[-1])
    ax_T.set_ylim(T_low, T_high)
    ax_T.set_xlabel("t (min)")
    ax_T.set_title("solid: RK4    dashed: exact exponential", fontsize=9)
    ax_T.legend(fontsize=7, loc="upper right", title="1/min", title_fontsize=7)

    def update(step: int):
        for b, (bar, line) in enumerate(zip(bars, lines)):
            bar.set_height(T[step, b] - T_low)
            bar.set_facecolor(cmap((T[step, b] - T_low) / (T_high - T_low)))
            line.set_data(t[: step + 1], T[: step + 1, b])
        clock.set_text(f"t = {t[step]:.0f} min")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
