"""Six views of the rotating mesh: the z-buffer, three shading models and perspective-correct vs affine texturing."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import frame_indices, save_gif

TITLES = [
    "z-buffer depth (dark = near)",
    "flat shading",
    "Gouraud (light corners, blend)",
    "Phong (blend normals, light)",
    "texture, perspective-correct",
    "texture, affine (wrong)",
]


def animate(frames: torch.Tensor, mesh: str, path: Path) -> Path:
    """frames (F, 6, H, W, 3) uint8."""
    imgs = frames.numpy()
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.0))
    shown = []
    for ax, title, img in zip(axes.flat, TITLES, imgs[0]):
        shown.append(ax.imshow(img, interpolation="nearest"))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_color("#d1d5db")
    heading = fig.suptitle("", fontsize=11)

    def update(f: int):
        for im, img in zip(shown, imgs[f]):
            im.set_data(img)
        heading.set_text(f"rasterised {mesh}: frame {f + 1}/{len(imgs)}")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_gif(fig, update, frame_indices(len(imgs), 120), path, fps=12)
