"""Animate the design: |E(x, z)| at both wavelengths, the meta-atom phases and the focal efficiency vs iteration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

GREY = "#6b7280"
COLORS = ["#16a34a", "#dc2626"]  # green and red for the two wavelengths


def animate(
    maps: torch.Tensor,
    map_hyp: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    snaps: torch.Tensor,
    theta_hyp: torch.Tensor,
    centres: torch.Tensor,
    history: torch.Tensor,
    eta_hyp: torch.Tensor,
    snap_iters: list[int],
    wavelengths: torch.Tensor,
    f: float,
    path: Path,
) -> Path:
    maps_np = maps.numpy()  # (S, L, Z, X)
    xs, zs = x.numpy(), z.numpy()
    lams = wavelengths.tolist()
    hist = history.numpy()  # (n_iter + 1, L)
    th = np.remainder(snaps.numpy(), 1.0)  # (S, n_atoms)
    xc = centres.numpy()
    vmax = [max(maps_np[:, li].max(), map_hyp[li].max().item()) for li in range(2)]

    fig = plt.figure(figsize=(8.4, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    axes_map = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    ax_p = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    cmap = plt.get_cmap("magma", 14)

    images = []
    for li, ax in enumerate(axes_map):
        im = ax.imshow(maps_np[0, li], origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax[li],
                       extent=[xs[0], xs[-1], zs[0], zs[-1]], interpolation="nearest")  # fmt: skip
        ax.plot([0], [f], "+", color="white", ms=10, mew=1.5)
        ax.set_title(f"|E(x, z)| at λ = {lams[li]:.3f} μm", fontsize=10, color=COLORS[li])
        ax.set_xlabel("x (μm)", fontsize=9)
        ax.grid(False)
        ax.tick_params(labelsize=8)
        images.append(im)
    axes_map[0].set_ylabel("z (μm)", fontsize=9)

    # meta-atom phases θ (in waves at λ0)
    ax_p.plot(xc, theta_hyp.numpy(), ".", color=GREY, ms=2.5, label="hyperbolic (wrapped)")
    (dots,) = ax_p.plot(xc, th[0], ".", color="#2563eb", ms=3.5, label="optimised")
    ax_p.set_ylim(-0.05, 1.3)
    ax_p.set_xlabel("x (μm)", fontsize=9)
    ax_p.set_ylabel("θ = φ / 2π at λ0", fontsize=9)
    ax_p.set_title("meta-atom phase delays", fontsize=10)
    ax_p.legend(loc="upper center", fontsize=7, ncol=2)
    ax_p.tick_params(labelsize=8)

    # focal-spot efficiency vs iteration
    it = np.arange(len(hist))
    ax_e.set_xlim(0, max(1, len(hist) - 1))
    ax_e.set_ylim(0, max(hist.max(), eta_hyp.max().item()) * 1.2)
    lines = []
    for li in range(2):
        ax_e.axhline(eta_hyp[li].item(), color=COLORS[li], ls="--", lw=1, alpha=0.8)
        lines.append(ax_e.plot([], [], color=COLORS[li], lw=1.8, label=f"{lams[li]:.3f} μm")[0])
    ax_e.plot([], [], ls="--", color=GREY, lw=1, label="hyperbolic")
    ax_e.set_xlabel("Adam iteration", fontsize=9)
    ax_e.set_title("power in focal spot η", fontsize=10)
    ax_e.legend(loc="lower right", fontsize=7)
    ax_e.tick_params(labelsize=8)
    label = ax_e.text(0.03, 0.94, "", transform=ax_e.transAxes, va="top", fontsize=8)

    def update(s: int):
        for li in range(2):
            images[li].set_data(maps_np[s, li])
            lines[li].set_data(it[: snap_iters[s] + 1], hist[: snap_iters[s] + 1, li])
        dots.set_ydata(th[s])
        eta = hist[snap_iters[s]]
        label.set_text(f"iteration {snap_iters[s]}:  η = {eta[0]:.0%} / {eta[1]:.0%}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(snaps), 60), path, fps=10)
