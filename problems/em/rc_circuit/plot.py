"""Animate the circuits: a bar per capacitor voltage, and the V(t) and I(t) curves growing in time."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(result: dict, R: torch.Tensor, C: float, path: Path) -> Path:
    t = result["t"].cpu().numpy()  # (T,)
    Vs = result["Vs"].cpu().numpy()  # (T,)
    V = result["V"].cpu().numpy()  # (T, B)
    I = result["I"].cpu().numpy()  # (T, B)  # noqa: E741
    labels = [f"{r:.2g}\nτ={r * C:.2g}s" for r in R.tolist()]  # bar labels: resistance and time constant
    top = max(Vs.max(), 1e-6) * 1.1

    fig = plt.figure(figsize=(7, 4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.6])
    ax_bar = fig.add_subplot(grid[:, 0])
    ax_V = fig.add_subplot(grid[0, 1])
    ax_I = fig.add_subplot(grid[1, 1], sharex=ax_V)

    # left: one bar per capacitor
    colours = [f"C{b}" for b in range(V.shape[1])]
    bars = ax_bar.bar(range(V.shape[1]), V[0], color=colours)
    source_line = ax_bar.axhline(Vs[0], color="k", ls="--", lw=1, label="source V_s")
    ax_bar.set_xticks(range(V.shape[1]), labels, fontsize=7)
    ax_bar.set_xlabel("R (kΩ)")
    ax_bar.set_ylabel("capacitor voltage V (V)")
    ax_bar.set_ylim(0, top)
    ax_bar.grid(axis="x", visible=False)
    ax_bar.legend(loc="upper right", fontsize=8)
    ax_bar.set_title("t = 0.0 s")  # placeholder so tight_layout leaves room for the clock

    # right: V(t) and I(t) for every circuit
    lines_V = [ax_V.plot([], [], color=c, lw=1.5)[0] for c in colours]
    lines_I = [ax_I.plot([], [], color=c, lw=1.5)[0] for c in colours]
    (source_curve,) = ax_V.plot([], [], color="k", ls="--", lw=1)
    ax_V.set_xlim(0, t[-1])
    ax_V.set_ylim(0, top)
    ax_V.set_ylabel("V (V)")
    ax_I.set_ylim(min(I.min(), 0) * 1.1, max(I.max(), 1e-6) * 1.1)
    ax_I.set_ylabel("I (mA)")
    ax_I.set_xlabel("t (s)")
    plt.setp(ax_V.get_xticklabels(), visible=False)

    def update(step: int):
        for b, bar in enumerate(bars):
            bar.set_height(V[step, b])
            lines_V[b].set_data(t[: step + 1], V[: step + 1, b])
            lines_I[b].set_data(t[: step + 1], I[: step + 1, b])
        source_line.set_ydata([Vs[step]] * 2)
        source_curve.set_data(t[: step + 1], Vs[: step + 1])
        ax_bar.set_title(f"t = {t[step]:.1f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path)
