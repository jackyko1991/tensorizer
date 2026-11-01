"""Animate aperture (left) and far-field log-intensity (right) side by side, cycling through the shapes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import airy_radius
from .tensors import N

VIEW = 48  # show the central ±VIEW pixels of both planes


def animate(schedule: list, masks: torch.Tensor, patterns: torch.Tensor, path: Path) -> Path:
    centre = slice(N // 2 - VIEW, N // 2 + VIEW)
    a = masks[:, centre, centre].numpy()  # (F, 2V, 2V)
    log_i = patterns[:, centre, centre].numpy()  # (F, 2V, 2V)
    extent = (-VIEW - 0.5, VIEW - 0.5, -VIEW - 0.5, VIEW - 0.5)

    fig, (ax_a, ax_i) = plt.subplots(1, 2, figsize=(7, 4))
    img_a = ax_a.imshow(a[0], origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    img_i = ax_i.imshow(log_i[0], origin="lower", extent=extent, cmap=plt.get_cmap("magma", 20), vmin=-5, vmax=0)
    fig.colorbar(img_i, ax=ax_i, fraction=0.046, pad=0.04, label="log₁₀ I / I_max")
    (ring,) = ax_i.plot([], [], "--", color="#22d3ee", lw=1)
    ax_a.set_xlabel("x (px)")
    ax_a.set_ylabel("y (px)")
    ax_i.set_xlabel("frequency l (px)")
    ax_i.set_ylabel("frequency k (px)")
    for ax in (ax_a, ax_i):
        ax.grid(False)
    title_a, title_i = ax_a.set_title(" "), ax_i.set_title("far field |U|²")
    circle = np.linspace(0, 2 * np.pi, 100)

    def update(f: int):
        shape, size = schedule[f]
        img_a.set_data(a[f])
        img_i.set_data(log_i[f])
        title_a.set_text(f"aperture: {shape}, D = {size:.0f} px")
        if shape == "circle":  # predicted first dark ring at 1.22 N / D
            r = airy_radius(N, size)
            ring.set_data(r * np.cos(circle), r * np.sin(circle))
            title_i.set_text("far field |U|²  (dashed: 1.22 N/D)")
        else:
            ring.set_data([], [])
            title_i.set_text("far field |U|²")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(schedule)), path, fps=8)
