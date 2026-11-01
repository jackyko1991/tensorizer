"""One static figure: smoothing and sharpening, edge detectors, noise removal and frequency responses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from problems._common.plotting import ACCENT, save_png


def figure(out: dict, responses: dict, p: dict, path: Path) -> Path:
    img = {k: v.cpu().numpy() for k, v in out.items()}
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.8))

    def show(ax, data, title, cmap="gray", vmin=0.0, vmax=1.0):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    show(axes[0, 0], img["noisy"], f"test image + noise σ = {p['noise']:g}")
    show(axes[0, 1], img["box"], f"box blur {p['box']}×{p['box']}")
    show(axes[0, 2], img["gauss"], f"Gaussian blur σ = {p['sigma']:g} (separable)")
    show(axes[0, 3], img["sharp"], f"unsharp mask, α = {p['amount']:g}")

    g = np.abs(img["gx"]).max()
    show(axes[1, 0], img["gx"], "Sobel $G_x$", cmap="RdBu_r", vmin=-g, vmax=g)
    show(axes[1, 1], img["mag"], r"$\sqrt{G_x^2 + G_y^2}$", cmap="gray_r", vmin=0, vmax=img["mag"].max())
    strength = np.clip(img["mag"] / (0.5 * img["mag"].max()), 0, 1)
    hue = (img["angle"] + np.pi) / (2 * np.pi)  # orientation → colour, brightness → edge strength
    show(axes[1, 2], hsv_to_rgb(np.stack([hue, np.ones_like(hue), strength], -1)), "orientation atan2$(G_y, G_x)$")
    lg = np.abs(img["log"]).max()
    show(axes[1, 3], img["log"], f"Laplacian of Gaussian σ = {p['sigma']:g}", cmap="RdBu_r", vmin=-lg, vmax=lg)

    show(axes[2, 0], img["sp"], f"salt & pepper {p['salt_pepper']:.0%}")
    show(axes[2, 1], img["sp_gauss"], "Gaussian on salt & pepper")
    show(axes[2, 2], img["sp_median"], f"median {p['median']}×{p['median']} on salt & pepper")

    ax = axes[2, 3]
    for (name, h), colour in zip(responses.items(), ACCENT):
        h = h.cpu().numpy()
        ax.plot(np.linspace(0, 1, len(h)), h, color=colour, lw=1.5, label=name)
    ax.axhline(1, color="#9ca3af", lw=0.8, ls="--")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_xlabel(r"horizontal frequency $k_x/\pi$", fontsize=8)
    ax.set_ylabel(r"$\hat H(k_x, 0)$", fontsize=8)
    ax.set_title("frequency responses", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left")

    fig.tight_layout()
    return save_png(fig, path)
