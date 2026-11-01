"""Side by side: |div u| (P waves) and |curl u| (S waves), with circles at the predicted fronts c (t - t0)."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from problems._common.plotting import frame_indices, save_gif


def animate(result: dict, c_p: float, c_s: float, t0: float, border: int, direction_deg: float, path: Path) -> Path:
    t = result["times"].cpu().numpy()
    fields = [result["div"].abs().cpu().numpy(), result["curl"].abs().cpu().numpy()]  # (T, N, N) each
    n = fields[0].shape[-1]
    centre = (n - 1) / 2
    arrow = 12 * np.array([math.cos(math.radians(direction_deg)), math.sin(math.radians(direction_deg))])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    images, circles = [], []
    for ax, field, speed, name, cmap in zip(axes, fields, (c_p, c_s), ("|div u|  (P wave)", "|curl u|  (S wave)"),
                                            ("Blues", "Reds")):  # fmt: skip
        top = np.quantile(field, 0.999) + 1e-12
        images.append(ax.imshow(field[0].T, origin="lower", cmap=plt.get_cmap(cmap, 10), vmin=0, vmax=top,
                                extent=(0, n, 0, n), interpolation="nearest"))  # fmt: skip
        circles.append(ax.add_patch(Circle((centre, centre), 0, fill=False, ec="#111827", ls="--", lw=1.0)))
        ax.add_patch(
            Rectangle((border, border), n - 2 * border, n - 2 * border, fill=False, ec="#9ca3af", ls=":", lw=0.8)
        )
        ax.annotate("", (centre + arrow[0], centre + arrow[1]), (centre - arrow[0], centre - arrow[1]),
                    arrowprops={"arrowstyle": "->", "color": "#16a34a", "lw": 1.5})  # fmt: skip
        ax.set_title(f"{name}   dashed: c t′, c = {speed:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    clock = fig.suptitle(" ", fontsize=10)

    def update(step: int):
        for image, circle, field, speed in zip(images, circles, fields, (c_p, c_s)):
            image.set_data(field[step].T)
            circle.set_radius(max(0.0, speed * (t[step] - t0)))  # t′ = t - t0: time since the wavelet peak
        clock.set_text(f"point force (green arrow) in an elastic solid   t = {t[step]:.1f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path, fps=12)
