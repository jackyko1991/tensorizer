"""Top: u(x) steepening into a shock, with the exact Cole–Hopf solution. Bottom: steepest slope vs time."""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(
    x: torch.Tensor, times: torch.Tensor, u: torch.Tensor, exact: Callable[[float], torch.Tensor],
    steepness: torch.Tensor, t_star: float, path: Path,
) -> Path:  # fmt: skip
    """exact(t) returns the Cole–Hopf solution at time t; it is evaluated only for the GIF frames."""
    xs, t, us, steep = x.cpu().numpy(), times.cpu().numpy(), u.cpu().numpy(), steepness.cpu().numpy()
    frames = frame_indices(len(t), 90)
    reference = {k: exact(float(t[k])).cpu().numpy() for k in frames}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 4.6), gridspec_kw={"height_ratios": [1.6, 1]})
    ax.plot(xs, us[0], color="#9ca3af", lw=1, label="u₀")
    line = ax.plot(xs, us[0], color="#2563eb", lw=2, label="finite differences")[0]
    ref = ax.plot(xs, us[0], color="#111827", lw=1, ls="--", label="Cole–Hopf (exact)")[0]
    pad = 0.15 * (us.max() - us.min())
    ax.set_xlim(0, xs[-1])
    ax.set_ylim(us.min() - pad, us.max() + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(loc="lower left", fontsize=8)
    title = ax.set_title(" ")

    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(0, steep.max() * 1.1)
    ax2.axvline(t_star, color="#f59e0b", ls="--", lw=1.2)
    ax2.text(t_star, steep.max() * 0.08, " t* = 1/max(−u₀′)", color="#b45309", fontsize=8, va="bottom")
    ax2.set_xlabel("t")
    ax2.set_ylabel("max(−u_x)")
    curve = ax2.plot([], [], color="#dc2626", lw=1.8)[0]

    def update(step: int):
        line.set_ydata(us[step])
        ref.set_ydata(reference[step])
        curve.set_data(t[: step + 1], steep[: step + 1])
        title.set_text(f"viscous Burgers   t = {t[step]:.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frames, path, fps=15)
