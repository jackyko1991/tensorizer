"""Top: the Venturi tube with tracers and three pressure gauges. Bottom: u(x) and p(x) along the axis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from . import tensors

GAUGES = [0.07, 0.46, 0.95]  # gauge positions along the axis (m): inlet, throat, outlet


def animate(
    x: torch.Tensor, flow: dict, times: torch.Tensor, tracer_x: torch.Tensor, eta: torch.Tensor, r1: float, r2: float,
    path: Path,
) -> Path:  # fmt: skip
    xs, r, u, p = (v.cpu().numpy() for v in (x, flow["r"], flow["u"], flow["p"]))
    t = times.cpu().numpy()
    px = tracer_x.cpu().numpy()  # (T, n)
    e = eta.cpu().numpy()  # (n,)
    p_kpa = p / 1000

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 4.6), gridspec_kw={"height_ratios": [1.5, 1]}, sharex=True)
    ax.fill_between(xs, -r, r, color="#dbeafe")
    ax.plot(xs, r, color="#374151", lw=2)
    ax.plot(xs, -r, color="#374151", lw=2)
    # gauges: vertical glass tubes whose liquid column is proportional to the local pressure
    top = 2.8 * r1 / max(p_kpa.max(), 1e-9)  # plot height per kPa
    for i, xg in enumerate(GAUGES):
        k = int(np.argmin(abs(xs - xg)))
        h = max(p_kpa[k], 0.0) * top
        ax.add_patch(plt.Rectangle((xg - 0.012, r[k]), 0.024, h, color="#2563eb", alpha=0.8))
        ax.plot([xg - 0.012, xg - 0.012, xg + 0.012, xg + 0.012], [r[k], r[k] + 3.2 * r1, r[k] + 3.2 * r1, r[k]],
                color="#9ca3af", lw=1)  # fmt: skip
        side = 1 if i < len(GAUGES) - 1 else -1  # the last label goes to the left of its tube
        ax.text(
            xg + side * 0.02,
            r[k] + h,
            f"{p_kpa[k]:.1f} kPa",
            fontsize=8,
            va="center",
            ha="left" if side > 0 else "right",
        )
    dots = ax.scatter(px[0], e * tensors.radius(torch.tensor(px[0]), r1, r2).numpy(), s=8, color="#dc2626", zorder=3)
    ax.set_ylim(-1.3 * r1, 4.5 * r1)
    ax.set_ylabel("y (m)")
    ax.grid(False)
    clock = ax.set_title("Q")  # placeholder text so tight_layout leaves room

    ax2.plot(xs, u, color="#dc2626", lw=1.8)
    ax2.set_ylabel("u (m/s)", color="#dc2626")
    ax2.set_xlabel("x (m)")
    ax3 = ax2.twinx()
    ax3.plot(xs, p_kpa, color="#2563eb", lw=1.8)
    ax3.set_ylabel("p (kPa)", color="#2563eb")
    ax3.grid(False)

    def update(step: int):
        y = e * tensors.radius(torch.tensor(px[step]), r1, r2).numpy()
        dots.set_offsets(np.column_stack([px[step], y]))
        clock.set_text(f"Q = {flow['Q'] * 1000:.2f} L/s    t = {t[step]:.2f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path, fps=20)
