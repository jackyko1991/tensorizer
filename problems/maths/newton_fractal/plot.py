"""Animate the basins of attraction filling in, a zoom onto their boundary, and the error of one starting point."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb

from problems._common.plotting import ACCENT, frame_indices, save_gif

GREY = np.array(to_rgb("#e5e7eb"))


def colour(labels: np.ndarray, conv: np.ndarray, k: int, n_roots: int) -> np.ndarray:
    """RGB image at step k: the root's colour, darker the more steps it took; grey while not yet converged."""
    base = np.array([to_rgb(ACCENT[j % len(ACCENT)]) for j in range(n_roots)])  # (J, 3)
    final = labels[-1].astype(int)
    shade = 1.0 - 0.75 * np.minimum(conv, 12)[..., None] / 12  # (H, W, 1)
    done = ((conv <= k) & (final >= 0))[..., None]
    return np.where(done, base[np.maximum(final, 0)] * shade, GREY)


def animate(views: list, n_roots: int, errors: torch.Tensor, limit: float, label: str, path: Path) -> Path:
    """views: [(labels (K+1, H, W), conv (H, W), half_width, center), ...] for the full plane and the zoom."""
    data = [(lab.cpu().numpy(), conv.cpu().numpy(), half, ctr) for lab, conv, half, ctr in views]
    n_steps = data[0][0].shape[0]
    e = errors.numpy()
    e = e[: int(np.argmax(e < 1e-15)) or len(e)]  # stop once the error reaches round-off

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    images = []
    for ax, (lab, conv, half, ctr), name in zip(axes, data, ["complex plane", "zoom onto a boundary point"]):
        ext = [ctr.real - half, ctr.real + half, ctr.imag - half, ctr.imag + half]
        images.append(ax.imshow(colour(lab, conv, 0, n_roots), extent=ext, interpolation="nearest"))
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Re z")
        ax.grid(False)
    axes[0].set_ylabel("Im z")
    zc, zh = data[1][3], data[1][2]
    axes[0].add_patch(plt.Rectangle((zc.real - zh, zc.imag - zh), 2 * zh, 2 * zh, fill=False, ec="#111827", lw=1.2))

    ax = axes[2]
    ks = np.arange(len(e))
    ax.semilogy(ks, e, "o-", color="#2563eb", ms=4, lw=1.2, label="error $e_k$")
    start = min(2, len(e) - 1)  # the squaring law holds once the point is close: start it at step 2
    guide = [e[start]]
    for _ in ks[start + 1 :]:
        guide.append(limit * guide[-1] ** 2)
    ax.semilogy(
        ks[start:], np.maximum(guide, 1e-17), "--", color="#9ca3af", lw=1, label=r"$e_{k+1} = |p''/2p'|\,e_k^2$"
    )
    ax.set_ylim(1e-16, 2)
    ax.set_xlabel("Newton step $k$")
    ax.set_title("quadratic convergence: digits double", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    (dot,) = ax.plot([], [], "o", color="#dc2626", ms=8)
    title = fig.suptitle("", fontsize=11)

    def update(k: int):
        for img, (lab, conv, _, _) in zip(images, data):
            img.set_data(colour(lab, conv, k, n_roots))
        j = min(k, len(e) - 1)
        dot.set_data([j], [e[j]])
        title.set_text(f"Newton's method for p(z) = {label}, step {k}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(n_steps, 120), path, fps=5)
