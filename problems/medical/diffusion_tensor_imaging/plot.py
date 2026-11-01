"""Diffusion ellipses on the FA map (left) and streamlines growing over the colour-FA map (right)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import EllipseCollection

from problems._common.plotting import frame_indices, save_gif


def animate(D: torch.Tensor, e1: torch.Tensor, fa: torch.Tensor, lines: torch.Tensor, n: int, path: Path) -> Path:
    fa_map = fa.reshape(n, n).clamp(0, 1).cpu().numpy()
    colour = (fa[:, None] * e1.abs()).clamp(0, 1).reshape(n, n, 3).cpu().numpy()  # red = x, green = y, blue = z
    lam2, vec2 = torch.linalg.eigh(D[:, :2, :2])  # in-plane 2×2 block → ellipse axes
    widths = torch.full_like(lam2[:, 1], 0.9).cpu().numpy()  # every glyph spans most of its voxel ...
    heights = (0.9 * lam2[:, 0] / lam2[:, 1].clamp(min=1e-9)).cpu().numpy()  # ... squashed by λ_small/λ_large
    angles = torch.rad2deg(torch.atan2(vec2[:, 1, 1], vec2[:, 0, 1])).cpu().numpy()
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    centres = np.stack([xx.ravel(), yy.ravel()], axis=1)
    tracks = lines.cpu().numpy()  # (S, L, 2)

    fig, (ax_g, ax_t) = plt.subplots(1, 2, figsize=(7.6, 3.9))
    ax_g.imshow(fa_map, cmap="gray", vmin=0, vmax=1, origin="lower")
    ax_g.add_collection(
        EllipseCollection(
            widths,
            heights,
            angles,
            units="xy",
            offsets=centres,
            offset_transform=ax_g.transData,
            facecolors=colour.reshape(-1, 3) * 0.7 + 0.3,
            edgecolors="#1f2937",
            linewidths=0.3,
        )
    )
    ax_g.set_title("diffusion ellipses on the FA map", fontsize=9)
    ax_t.imshow(colour, origin="lower")
    ax_t.set_title("colour FA (red x, green y) + tractography", fontsize=9)
    curves = [ax_t.plot([], [], color="white", lw=1.4)[0] for _ in range(tracks.shape[1])]
    heads = ax_t.scatter(np.zeros(tracks.shape[1]), np.zeros(tracks.shape[1]), s=8, c="#facc15", zorder=3)
    for ax in (ax_g, ax_t):
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    def update(step: int):
        for line, curve in zip(np.transpose(tracks[: step + 1], (1, 0, 2)), curves):
            curve.set_data(line[:, 0], line[:, 1])
        tips = np.array([line[~np.isnan(line[:, 0])][-1] for line in np.transpose(tracks[: step + 1], (1, 0, 2))])
        heads.set_offsets(tips)

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(tracks), 60), path, fps=12)
