"""Top: vorticity of the wake behind the cylinder. Bottom: drag and lift coefficients over time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

from problems._common.plotting import frame_indices, save_gif


def animate(
    snapshots: torch.Tensor, snapshot_steps: torch.Tensor, coeffs: torch.Tensor, centre: tuple[float, float],
    radius: float, u_in: float, path: Path,
) -> Path:  # fmt: skip
    w = snapshots.cpu().numpy()  # (F, Nx, Ny), NaN inside the cylinder
    steps = snapshot_steps.cpu().numpy()
    c = coeffs.cpu().numpy()  # (T, 2)
    nx, ny = w.shape[1:]
    limit = 2.5 * u_in / (2 * radius)  # vorticity scale U/D
    late = c[len(c) // 2 :]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.4), gridspec_kw={"height_ratios": [1.3, 1]})
    cmap = plt.get_cmap("RdBu_r", 15).with_extremes(bad="#9ca3af")  # grey inside the cylinder (NaN)
    image = ax.imshow(w[0].T, origin="lower", cmap=cmap, vmin=-limit, vmax=limit, extent=(0, nx, 0, ny),
                      interpolation="nearest")  # fmt: skip
    ax.add_patch(Circle(centre, radius, fc="#9ca3af", ec="#374151", lw=1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    title = ax.set_title(" ", fontsize=10)

    ax2.set_xlim(0, len(c))
    ax2.set_ylim(min(-1.2, 1.5 * late[:, 1].min()), max(3.0, 1.3 * late[:, 0].max()))
    ax2.axhline(0, color="#9ca3af", lw=0.8)
    ax2.set_xlabel("lattice time step")
    drag = ax2.plot([], [], color="#dc2626", lw=1.4, label="drag C_D")[0]
    lift = ax2.plot([], [], color="#2563eb", lw=1.4, label="lift C_L")[0]
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    t = np.arange(len(c))

    def update(k: int):
        image.set_data(w[k].T)
        n = int(steps[k])
        drag.set_data(t[: n + 1], c[: n + 1, 0])
        lift.set_data(t[: n + 1], c[: n + 1, 1])
        title.set_text(f"vorticity   step {n}   C_D = {c[max(0, n - 200) : n + 1, 0].mean():.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(steps), 90), path, fps=15)
