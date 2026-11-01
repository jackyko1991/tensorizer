"""Animate the spin lattice at the chosen temperature beside |m| and E/N for every lattice in the batch."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import onsager_magnetization


def animate(spins: torch.Tensor, m: torch.Tensor, E: torch.Tensor, T: torch.Tensor, path: Path) -> Path:
    s = spins[:, 0].cpu().numpy()  # (S+1, L, L)  lattice at the chosen temperature
    m_abs = m.abs().cpu().numpy()  # (S+1, B)
    E_np = E.cpu().numpy()  # (S+1, B)
    temps = T.cpu().tolist()
    exact = onsager_magnetization(T).cpu().tolist()
    sweeps = np.arange(len(s))
    labels = [f"T = {temps[0]:.2f} (chosen)", f"T = {temps[1]:.2f} = T_c", f"T = {temps[2]:.2f}"]

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax_m = fig.add_subplot(2, 2, 2)
    ax_E = fig.add_subplot(2, 2, 4, sharex=ax_m)

    image = ax.imshow(s[0], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    title = ax.set_title("")

    lines_m, lines_E = [], []
    for b, label in enumerate(labels):
        lines_m.append(ax_m.plot([], [], color=ACCENT[b], lw=1.5, label=label)[0])
        lines_E.append(ax_E.plot([], [], color=ACCENT[b], lw=1.5)[0])
    ax_m.axhline(exact[0], color=ACCENT[0], ls=":", lw=1)  # Onsager's |m| for the chosen T
    ax_m.set_xlim(0, sweeps[-1])
    ax_m.set_ylim(0, 1.05)
    ax_m.set_ylabel("|m|")
    ax_m.legend(fontsize=7, loc="center right")
    ax_m.set_title("dotted: Onsager's exact |m|", fontsize=9)
    ax_E.set_ylim(-2.05, 0.05)
    ax_E.set_xlabel("Monte Carlo sweeps")
    ax_E.set_ylabel("E / N")
    plt.setp(ax_m.get_xticklabels(), visible=False)

    def update(step: int):
        image.set_data(s[step])
        for b in range(len(labels)):
            lines_m[b].set_data(sweeps[: step + 1], m_abs[: step + 1, b])
            lines_E[b].set_data(sweeps[: step + 1], E_np[: step + 1, b])
        title.set_text(f"T = {temps[0]:.2f}   sweep {step}   m = {m_abs[step, 0]:.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(s), 100), path)
