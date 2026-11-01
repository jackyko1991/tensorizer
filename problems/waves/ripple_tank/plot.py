"""Left: the ripples. Right: the running intensity ⟨u²⟩, where the nodal lines appear dark. Dashed: predicted nodes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif


def animate(times, u, intensity, path_diff, s1, s2, wavelength: float, border: int, path: Path) -> Path:
    t = times.cpu().numpy()
    inner = slice(border, u.shape[-1] - border)  # hide the absorbing border
    U = u[:, inner, inner].cpu().numpy()  # (F, m, m)
    W = intensity[:, inner, inner].cpu().numpy()  # running ⟨u²⟩
    D = path_diff[inner, inner].cpu().numpy()
    lo, hi = border, u.shape[-1] - border
    extent = (lo, hi, lo, hi)
    u_lim = np.quantile(np.abs(U[-1]), 0.98) + 1e-9
    I_lim = np.quantile(W[-1], 0.98) + 1e-9
    m_max = int(D.max() / wavelength - 0.5)
    levels = [(m + 0.5) * wavelength for m in range(m_max + 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.7))
    im1 = ax1.imshow(
        U[0].T,
        origin="lower",
        extent=extent,
        cmap=plt.get_cmap("RdBu_r", 10),
        vmin=-u_lim,
        vmax=u_lim,
        interpolation="nearest",
    )
    im2 = ax2.imshow(
        W[0].T,
        origin="lower",
        extent=extent,
        cmap=plt.get_cmap("Blues", 8),
        vmin=0,
        vmax=I_lim,
        interpolation="nearest",
    )
    for ax, name in ((ax1, "surface u"), (ax2, "intensity ⟨u²⟩")):
        if levels:
            ax.contour(
                np.arange(lo, hi) + 0.5, np.arange(lo, hi) + 0.5, D.T, levels=levels, colors="#f59e0b",
                linestyles="--", linewidths=0.9,
            )  # fmt: skip
        ax.plot([s1[0], s2[0]], [s1[1], s2[1]], "o", color="#111827", ms=4)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    title = fig.suptitle("two coherent sources   t = 0", fontsize=11)

    def update(step: int):
        im1.set_data(U[step].T)
        im2.set_data(W[step].T)
        title.set_text(f"two coherent sources   λ = {wavelength:g}   t = {t[step]:.0f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 60), path, fps=10)
