"""Top: the wave now, with dotted 'ghost' solitons that ignore each other. Bottom: the x–t diagram filling in."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(times: torch.Tensor, x: torch.Tensor, u: torch.Tensor, ghosts: torch.Tensor, path: Path) -> Path:
    t, xs = times.cpu().numpy(), x.cpu().numpy()
    U, G = u.cpu().numpy(), ghosts.cpu().numpy()  # (F, J) each
    box = xs[-1] + (xs[1] - xs[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.8), gridspec_kw={"height_ratios": [1, 1.3]})
    ghost = ax1.plot(xs, G[0], ":", color="#9ca3af", lw=1.5, label="no interaction")[0]
    line = ax1.plot(xs, U[0], color="#2563eb", lw=2, label="KdV")[0]
    ax1.set_xlim(0, box)
    ax1.set_ylim(-0.1 * U.max(), 1.2 * U.max())
    ax1.set_ylabel("u")
    ax1.legend(loc="upper right", fontsize=8)
    title = ax1.set_title("KdV two-soliton collision   t = 0.0")

    shown = np.full_like(U, np.nan)
    image = ax2.imshow(
        shown, origin="lower", aspect="auto", extent=(0, box, t[0], t[-1]), cmap=plt.get_cmap("Blues", 12),
        vmin=0, vmax=0.7 * U.max(), interpolation="nearest",
    )  # fmt: skip
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.grid(False)

    def update(step: int):
        line.set_ydata(U[step])
        ghost.set_ydata(G[step])
        shown[: step + 1] = U[: step + 1]
        image.set_data(shown)
        title.set_text(f"KdV two-soliton collision   t = {t[step]:.1f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path, fps=12)
