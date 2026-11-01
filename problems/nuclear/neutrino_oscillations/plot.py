"""Top: flavour probabilities along the baseline (stacked). Bottom: a beam of neutrinos coloured by flavour."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

COLOURS = ["#2563eb", "#dc2626", "#16a34a"]  # e, μ, τ
NAMES = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]


def animate(
    dist: torch.Tensor,
    p: torch.Tensor,
    beam_l: torch.Tensor,
    beam_y: torch.Tensor,
    beam_flavour: torch.Tensor,
    born: int,
    energy: float,
    path: Path,
) -> Path:
    dist, p = dist.cpu().numpy(), p.cpu().numpy()  # (K,), (K, 3)
    bl, by, bf = beam_l.cpu().numpy(), beam_y.cpu().numpy(), beam_flavour.cpu().numpy()
    stacked = np.concatenate([np.zeros((len(dist), 1)), np.cumsum(p, axis=1)], axis=1)  # (K, 4) band edges

    fig, (ax_p, ax_b) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for beta in range(3):
        ax_p.fill_between(dist, stacked[:, beta], stacked[:, beta + 1], color=COLOURS[beta], alpha=0.12, lw=0)
    bands = []
    ax_p.set_xscale("log")
    ax_p.set_xlim(dist[0], dist[-1])
    ax_p.set_ylim(0, 1)
    ax_p.set_ylabel("flavour probability")
    ax_p.set_title(f"beam born as {NAMES[born]},  E = {energy:g} GeV", fontsize=10)
    front_p = ax_p.axvline(dist[0], color="#111827", lw=1)
    readout = ax_p.text(1.01, 0.5, "", transform=ax_p.transAxes, va="center", fontsize=9)
    for beta in range(3):
        ax_p.plot([], [], color=COLOURS[beta], lw=6, label=NAMES[beta])
    ax_p.legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.9)

    ax_b.set_ylim(-0.1, 1.1)
    ax_b.set_yticks([])
    ax_b.set_xlabel("distance travelled L (km)")
    ax_b.grid(False)
    scatter = ax_b.scatter([], [], s=12)
    front_b = ax_b.axvline(dist[0], color="#111827", lw=1)

    def update(step: int):
        for band in bands:
            band.remove()
        bands.clear()
        for beta in range(3):
            fill = ax_p.fill_between(
                dist[: step + 1], stacked[: step + 1, beta], stacked[: step + 1, beta + 1], color=COLOURS[beta], lw=0
            )
            bands.append(fill)
        front_p.set_xdata([dist[step]])
        front_b.set_xdata([dist[step]])
        readout.set_text("\n".join(f"{NAMES[b]} {p[step, b]:.2f}" for b in range(3)))
        shown = bl <= dist[step]
        scatter.set_offsets(np.column_stack([bl[shown], by[shown]]))
        scatter.set_color([COLOURS[f] for f in bf[shown]])

    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    return save_gif(fig, update, frame_indices(len(dist), 72), path)
