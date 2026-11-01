"""Top: |ψ(x)| now, with the Peregrine level 3. Bottom: the x–t map of |ψ| filling in as time runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(times: torch.Tensor, x: torch.Tensor, psi: torch.Tensor, path: Path) -> Path:
    t, xs = times.cpu().numpy(), x.cpu().numpy()
    amp = psi.abs().cpu().numpy()  # (F, J)
    running_max = np.maximum.accumulate(amp.max(axis=1))  # (F,)
    top = max(3.3, 1.1 * amp.max())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 4.8), gridspec_kw={"height_ratios": [1, 1.3]}, layout="constrained"
    )
    ax1.axhline(3.0, ls="--", color="#f59e0b", lw=1)
    line = ax1.plot(xs, amp[0], color="#2563eb", lw=1.5)[0]
    info = ax1.text(0.01, 0.95, "", transform=ax1.transAxes, va="top", fontsize=9)
    ax1.set_xlim(xs[0], xs[-1])
    ax1.set_ylim(0, top)
    ax1.set_ylabel("|ψ|")
    title = ax1.set_title("focusing NLS   t = 0.0")

    shown = np.full_like(amp, np.nan)
    image = ax2.imshow(
        shown, origin="lower", aspect="auto", extent=(xs[0], xs[-1], t[0], t[-1]), cmap=plt.get_cmap("magma_r", 12),
        vmin=0, vmax=top, interpolation="nearest",
    )  # fmt: skip
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.grid(False)
    fig.colorbar(image, ax=[ax1, ax2], label="|ψ|", pad=0.01, shrink=0.6, anchor=(0, 0))  # both panels keep one x-axis

    def update(step: int):
        line.set_ydata(amp[step])
        shown[: step + 1] = amp[: step + 1]
        image.set_data(shown)
        info.set_text(f"highest |ψ| so far: {running_max[step]:.2f}")
        title.set_text(f"focusing NLS   t = {t[step]:.1f}")

    return save_gif(fig, update, frame_indices(len(t), 90), path, fps=12)
