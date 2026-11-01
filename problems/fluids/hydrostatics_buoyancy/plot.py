"""Left: the tank coloured by pressure with the blocks bobbing. Right: submerged fraction of each block vs time."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    times: torch.Tensor,
    traj: torch.Tensor,
    rho_b: torch.Tensor,
    p: torch.Tensor,
    width: float,
    depth: float,
    s: float,
    rho_w: float,
    path: Path,
) -> Path:
    t = times.cpu().numpy()  # (T,)
    z = traj[:, :, 0].cpu().numpy()  # (T, B) height of each block's bottom face
    rho = rho_b.cpu().numpy()  # (B,)
    fraction = (-z).clip(0, s) / s  # (T, B)
    n = len(rho)
    gap = width / n
    colors = [ACCENT[b % len(ACCENT)] for b in range(n)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    gauge = (p - p[-1, 0]).cpu().numpy() / 1000  # (nz, nx) gauge pressure p - p0 in kPa
    image = ax.imshow(
        gauge, origin="lower", extent=(0, width, -depth, 0), cmap=plt.get_cmap("Blues", 8), vmin=0,
        vmax=gauge.max() * 1.15, aspect="auto",
    )  # fmt: skip
    fig.colorbar(image, ax=ax, label="p − p₀ (kPa)", shrink=0.8)
    ax.axhline(0, color="#1e40af", lw=1.5)
    ax.set_xlim(0, width)
    ax.set_ylim(-depth, 0.45)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.grid(False)
    blocks = [
        ax.add_patch(Rectangle(((b + 0.5) * gap - s / 2, z[0, b]), s, s, fc=colors[b], ec="#374151", lw=1))
        for b in range(n)
    ]
    labels = [ax.text((b + 0.5) * gap, 0.0, f"{rho[b]:.0f}", ha="center", fontsize=7) for b in range(n)]
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)

    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("submerged fraction")
    ax2.set_title("dashed: Archimedes ρ_b/ρ_w", fontsize=9)
    for b in range(n):
        ax2.axhline(min(1.0, rho[b] / rho_w), color=colors[b], ls="--", lw=0.8)
    curves = [ax2.plot([], [], color=colors[b], lw=1.6)[0] for b in range(n)]

    def update(step: int):
        for b in range(n):
            blocks[b].set_y(z[step, b])
            labels[b].set_y(z[step, b] + s + 0.03)
            curves[b].set_data(t[: step + 1], fraction[: step + 1, b])
        clock.set_text(f"t = {t[step]:.2f} s   (labels: ρ in kg/m³)")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path, fps=16)
