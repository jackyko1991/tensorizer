"""Animate the rendered image next to a top-down map of the scene, the camera and the light."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

from problems._common.plotting import save_gif


def animate(images: torch.Tensor, eyes: torch.Tensor, lights: torch.Tensor, sc: dict, orbit: str, path: Path) -> Path:
    """images (F, H, W, 3) in [0, 1], eyes and lights (F, 3)."""
    img = (images.cpu().numpy() * 255).astype(np.uint8)
    eye, light = eyes.cpu().numpy(), lights.cpu().numpy()

    fig, (ax_i, ax_m) = plt.subplots(1, 2, figsize=(8.4, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    image = ax_i.imshow(img[0], interpolation="bilinear")
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    ax_i.grid(False)
    title = ax_i.set_title("", fontsize=10)

    for c, r, a in zip(sc["centres"].cpu().numpy(), sc["radii"].cpu().numpy(), sc["albedo"].cpu().numpy()):
        ax_m.add_patch(Circle((c[0], c[2]), r, facecolor=a, edgecolor="#374151", lw=1))
    angle = np.linspace(0, 2 * math.pi, 100)
    moving = light if orbit == "light" else eye
    radius = np.hypot(moving[0, 0], moving[0, 2])
    ax_m.plot(radius * np.sin(angle), radius * np.cos(angle), "--", color="#9ca3af", lw=1)
    (cam,) = ax_m.plot([], [], "s", color="#374151", ms=8, label="camera")
    (sight,) = ax_m.plot([], [], "-", color="#374151", lw=1)
    (lamp,) = ax_m.plot([], [], "*", color="#f59e0b", ms=15, mec="#b45309", label="light")
    ax_m.set_xlim(-8.5, 8.5)
    ax_m.set_ylim(8.5, -8.5)  # +z points towards the viewer at the bottom
    ax_m.set_aspect("equal")
    ax_m.set_xlabel("x")
    ax_m.set_ylabel("z")
    ax_m.set_title("top view", fontsize=10)
    ax_m.legend(loc="upper right", fontsize=8)

    def update(f: int):
        image.set_data(img[f])
        cam.set_data([eye[f, 0]], [eye[f, 2]])
        sight.set_data([eye[f, 0], 0], [eye[f, 2], 0])
        lamp.set_data([light[f, 0]], [light[f, 2]])
        title.set_text(f"ray traced, {orbit} orbiting — frame {f + 1}/{len(img)}")

    fig.tight_layout()
    return save_gif(fig, update, list(range(len(img))), path, fps=12)
