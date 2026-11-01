"""Top: light intensity |ψ|² on the topological and the trivial flake. Bottom: Berry curvature, ribbon bands
with edge states, and the fraction of light that has passed the notch."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from problems._common.plotting import frame_indices, save_gif

LIGHT = LinearSegmentedColormap.from_list("light", ["#e5e7eb", "#60a5fa", "#1e3a8a"], N=8)  # discrete: small GIF


def animate(
    pos: torch.Tensor, intensity: torch.Tensor, times: torch.Tensor, transmitted: torch.Tensor, topo: dict,
    C_trivial: float, bands: dict, bands_triv: dict, M: float, M_trivial: float, notch: torch.Tensor, R: float,
    path: Path,
) -> Path:  # fmt: skip
    xy, inten, t, tr = pos.cpu().numpy(), intensity.cpu().numpy(), times.cpu().numpy(), transmitted.cpu().numpy()
    peak = 0.4 * inten[0, 0].max()  # colour scale fixed by the input light

    fig = plt.figure(figsize=(7.6, 5.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    top = grid[0, :].subgridspec(1, 2, wspace=0.05)
    names = [f"M = {M:.2f}:  C = {round(topo['C']):+d}", f"M = {M_trivial:.2f}:  C = {round(C_trivial):+d} (trivial)"]
    dots, flakes = [], []
    for s in range(2):
        ax = fig.add_subplot(top[0, s])
        dots.append(ax.scatter(xy[:, 0], xy[:, 1], s=9, c=inten[s, 0], cmap=LIGHT, vmin=0, vmax=peak, lw=0))
        ax.add_patch(plt.Circle(notch.cpu().numpy(), R, fill=False, ls="--", color="#9ca3af", lw=1))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(names[s], fontsize=9)
        flakes.append(ax)
    clock = flakes[0].text(0.01, 0.98, "", transform=flakes[0].transAxes, va="top", fontsize=8)

    # Berry curvature of the lower band on the k-grid (reduced coordinates k_1, k_2 ∈ [0, 1))
    ax = fig.add_subplot(grid[1, 0])
    om = topo["omega"].cpu().numpy()
    lim = max(np.abs(om).max(), 1e-12)
    ax.imshow(om.T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xlabel("k₁ / |G₁|")
    ax.set_ylabel("k₂ / |G₂|")
    ax.grid(False)
    ax.set_title(f"Berry curvature Ω,  C = {topo['C']:+.2f}", fontsize=8)

    # ribbon bands: light grey = trivial lattice; dark = topological, with red/blue = bottom/top edge states
    ax = fig.add_subplot(grid[1, 1])
    kx = bands["kx"].cpu().numpy() * np.sqrt(3)
    E, bottom, top_w = bands["E"].cpu().numpy(), bands["bottom"].cpu().numpy(), bands["top"].cpu().numpy()
    ax.plot(kx, bands_triv["E"].cpu().numpy(), color="#d1d5db", lw=0.6)
    ax.plot(kx, E, color="#4b5563", lw=0.5)
    kk = np.repeat(kx[:, None], E.shape[1], 1)
    ax.plot(kk[bottom > 0.6], E[bottom > 0.6], ".", color="#dc2626", ms=2)
    ax.plot(kk[top_w > 0.6], E[top_w > 0.6], ".", color="#2563eb", ms=2)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("k_x a√3")
    ax.set_ylabel("detuning E")
    ax.set_title("ribbon: bottom / top edge", fontsize=8)

    # fraction of the light that has passed the notch and still runs along the edge
    ax = fig.add_subplot(grid[1, 2])
    ax.plot(t, tr[0], color="#2563eb", lw=1.8, label="topological")
    ax.plot(t, tr[1], color="#9ca3af", lw=1.8, label="trivial")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time t")
    ax.set_ylabel("edge light past notch")
    ax.legend(fontsize=7, loc="upper left")
    cursor = ax.axvline(0, color="#f59e0b", lw=1.2)

    def update(step: int):
        for s in range(2):
            dots[s].set_array(inten[s, step])
        cursor.set_xdata([t[step]])
        clock.set_text(f"t = {t[step]:.1f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 60), path, fps=10)
