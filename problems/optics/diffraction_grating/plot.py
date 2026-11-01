"""Animate the grating pattern as N grows, next to the chain of phasors at one angle."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import wavenumber


def animate(
    sin_theta: torch.Tensor,
    patterns: torch.Tensor,
    env: torch.Tensor,
    counts: list[int],
    frames: list,
    orders: list[float],
    wavelength: float,
    d: float,
    path: Path,
) -> Path:
    x = sin_theta.numpy()  # (T,)
    intensity = patterns.numpy()  # (len(counts), T)
    k = wavenumber(wavelength)

    fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [2, 1]})
    for s in orders:  # d sin θ = m λ
        ax.axvline(s, color="#9ca3af", lw=0.8, ls=":")
    ax.plot(x, env.numpy(), "--", color="#6b7280", lw=1, label="single-slit envelope")
    (curve,) = ax.plot([], [], color="#2563eb", lw=1.2)
    (marker,) = ax.plot([], [], "o", color="#dc2626", ms=5)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("sin θ")
    ax.set_ylabel("intensity I / I(0)")
    ax.legend(loc="upper right", fontsize=7)
    title = ax.set_title(" ")

    ax_p.set_aspect("equal")
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    (chain,) = ax_p.plot([], [], "-o", color="#2563eb", ms=2.5, lw=1.2)
    (total,) = ax_p.plot([], [], color="#dc2626", lw=2)
    label = ax_p.set_title(" ")

    def update(f: int):
        i, phi, tips = frames[f]
        n = counts[i]
        curve.set_data(x, intensity[i])
        s = phi / (k * d)  # the angle this phase step belongs to: φ = k d sin θ
        marker.set_data([s], [np.interp(s, x, intensity[i])])
        title.set_text(f"N = {n} slits")
        z = tips.numpy()
        chain.set_data(z.real, z.imag)
        total.set_data([0, z[-1].real], [0, z[-1].imag])  # the resultant N·E
        ax_p.set_xlim(-0.3 * n, 1.1 * n)
        ax_p.set_ylim(-0.35 * n, 0.6 * n)
        label.set_text(f"phasors, Nφ = {n * phi / np.pi:.1f}π")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(frames)), path, fps=8)
