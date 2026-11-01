"""Left: vorticity with streamlines (contours of ψ) and the vortex centre. Right: u on the vertical centreline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(
    y: torch.Tensor, times: torch.Tensor, omega: torch.Tensor, psi: torch.Tensor, u_centre: torch.Tensor,
    ghia: torch.Tensor | None, re: float, path: Path,
) -> Path:  # fmt: skip
    ys, t = y.cpu().numpy(), times.cpu().numpy()
    w, p, uc = omega.cpu().numpy(), psi.cpu().numpy(), u_centre.cpu().numpy()  # (F, N, N), (F, N, N), (F, N)
    levels = np.concatenate([-np.geomspace(1e-4, 0.1, 9)[::-1], np.geomspace(1e-6, 1e-4, 3)])  # ψ < 0: main vortex

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    image = ax.imshow(
        w[0].T, origin="lower", extent=(0, 1, 0, 1), cmap=plt.get_cmap("RdBu_r", 15), vmin=-6, vmax=6,
        interpolation="nearest",
    )  # fmt: skip
    fig.colorbar(image, ax=ax, label="vorticity ω", shrink=0.8)
    ax.annotate("", (0.85, 1.04), (0.15, 1.04), arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 2},
                annotation_clip=False)  # fmt: skip
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)
    centre = ax.plot([], [], "x", color="#f59e0b", ms=9, mew=2)[0]
    title = ax.set_title(" ", pad=16)
    contours = []

    ax2.plot([0, 0], [0, 1], color="#9ca3af", lw=0.8)
    if ghia is not None:
        g = ghia.cpu().numpy()
        ax2.plot(g[:, 1], g[:, 0], "o", mfc="none", color="#111827", ms=5, label="Ghia et al. 1982")
    line = ax2.plot(uc[0], ys, color="#2563eb", lw=2, label="this solver")[0]
    ax2.set_xlim(-0.5, 1.05)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("u(x = 0.5, y)")
    ax2.set_ylabel("y")
    ax2.legend(loc="lower right", fontsize=8)

    def update(step: int):
        image.set_data(w[step].T)
        for c in contours:
            c.remove()
        contours.clear()
        contours.append(ax.contour(ys, ys, p[step].T, levels=levels, colors="k", linewidths=0.6))
        i, j = np.unravel_index(np.argmin(p[step]), p[step].shape)
        centre.set_data([ys[i]], [ys[j]])
        line.set_xdata(uc[step])
        title.set_text(f"Re = {re:.0f}   t = {t[step]:.1f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path, fps=12)
