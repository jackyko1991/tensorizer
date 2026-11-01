"""Top: the shower growing in depth (e± solid, γ dashed). Bottom: particle count against depth."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from problems._common.plotting import frame_indices, save_gif


def animate(
    depths: torch.Tensor,
    result: dict[str, torch.Tensor],
    reference: torch.Tensor,
    reference_label: str,
    n_max: float,
    t_max: float,
    path: Path,
) -> Path:
    t = depths.cpu().numpy()
    counts = result["counts"].cpu().numpy()  # (T, 2) photons, e±
    total = counts.sum(axis=1)
    ref = reference.cpu().numpy()  # (T,) total count with the other splitting mode
    z0, x0, z1, x1, kind = result["tracks"].cpu().numpy().T

    fig, (ax_s, ax_n) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    lateral = np.percentile(np.abs(np.concatenate([x0, x1])), 99.5) * 1.1 + 1e-3
    ax_s.set_xlim(0, t[-1])
    ax_s.set_ylim(-lateral, lateral)
    ax_s.set_ylabel("lateral (X₀)")
    ax_s.grid(False)
    electrons = LineCollection([], colors="#2563eb", linewidths=0.6, alpha=0.7)
    photons = LineCollection([], colors="#f59e0b", linewidths=0.6, alpha=0.7, linestyles=(0, (2, 2)))
    ax_s.add_collection(photons)
    ax_s.add_collection(electrons)
    ax_s.plot([], [], color="#2563eb", label="e±")
    ax_s.plot([], [], color="#f59e0b", ls="--", label="γ")
    ax_s.legend(loc="upper left", fontsize=8)
    front_s = ax_s.axvline(0, color="#111827", lw=1)
    title = ax_s.set_title(" ", fontsize=9)  # reserve room before tight_layout

    ax_n.plot(t, ref, color="#9ca3af", lw=1.5, ls="--", label=reference_label)
    ax_n.axhline(n_max, color="#f59e0b", lw=1, ls=":")
    ax_n.text(t[-1], n_max, r"$E_0/E_c$ ", ha="right", va="bottom", fontsize=8, color="#b45309")
    ax_n.axvline(t_max, color="#f59e0b", lw=1, ls=":")
    ax_n.text(t_max, 0.02 * n_max, r" $\ln(E_0/E_c)/\ln 2$", fontsize=8, color="#b45309")
    (line_tot,) = ax_n.plot([], [], color="#111827", lw=2, label="all particles")
    (line_e,) = ax_n.plot([], [], color="#2563eb", lw=1.2, label="e±")
    (line_g,) = ax_n.plot([], [], color="#f59e0b", lw=1.2, label="γ")
    ax_n.set_ylim(0, max(total.max(), ref.max(), n_max) * 1.12)
    ax_n.set_xlabel("depth t (radiation lengths X₀)")
    ax_n.set_ylabel("particles in flight")
    ax_n.legend(loc="upper right", fontsize=7)

    def clipped(front: float, which: np.ndarray) -> np.ndarray:
        sel = which & (z0 < front)
        end = np.minimum(z1[sel], front)
        frac = (end - z0[sel]) / np.maximum(z1[sel] - z0[sel], 1e-12)
        x_end = x0[sel] + frac * (x1[sel] - x0[sel])
        return np.stack([np.column_stack([z0[sel], x0[sel]]), np.column_stack([end, x_end])], axis=1)

    def update(step: int):
        front = t[step]
        electrons.set_segments(clipped(front, kind == 1))
        photons.set_segments(clipped(front, kind == 0))
        front_s.set_xdata([front])
        line_tot.set_data(t[: step + 1], total[: step + 1])
        line_e.set_data(t[: step + 1], counts[: step + 1, 1])
        line_g.set_data(t[: step + 1], counts[: step + 1, 0])
        title.set_text(f"depth {front:.1f} X₀   particles in flight: {total[step]}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path)
