"""Static figure: potential heatmap with equipotentials, field streamlines and the charges."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import save_png


def draw(x: torch.Tensor, phi: torch.Tensor, E: torch.Tensor, rho: torch.Tensor, path: Path) -> Path:
    xs = x.numpy()  # (N,) same coordinates for x and y
    p = phi.numpy()  # (N, N) rows = y, columns = x
    E_x, E_y = E.numpy()
    limit = np.quantile(np.abs(p), 0.98)  # φ ~ log r blows up at a line charge: clip the colour scale

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        p, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="bilinear"
    )
    ax.contour(xs, xs, p, levels=np.linspace(-limit, limit, 13), colors="k", linewidths=0.6, alpha=0.6)
    ax.streamplot(xs, xs, E_x, E_y, color="#374151", density=1.1, linewidth=0.7, arrowsize=0.8)

    a, b = np.nonzero(rho.numpy())  # cells that carry charge
    sign = np.sign(rho.numpy()[a, b])
    ax.scatter(
        xs[b], xs[a], c=np.where(sign > 0, "#dc2626", "#2563eb"), s=max(8, 60 / len(a)), edgecolors="k", zorder=3
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("potential φ, equipotentials and field lines E = −∇φ")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="φ")
    fig.tight_layout()
    return save_png(fig, path)
