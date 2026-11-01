"""Animate FRAP: the bleached spot filling in on three membranes, and the recovery curves with the Soumpasis fit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import save_gif

from .equation import soumpasis

ZOOM = 3.0  # show ±3 w around the spot


def animate(
    times: torch.Tensor,
    f: torch.Tensor,
    snap_times: torch.Tensor,
    snaps: torch.Tensor,
    x: torch.Tensor,
    w: float,
    depth: float,
    D: torch.Tensor,
    D_fit: torch.Tensor,
    t_half: torch.Tensor,
    names: list[str],
    path: Path,
) -> Path:
    keep = (x.abs() <= ZOOM * w).cpu().numpy()  # grid points inside the zoom window
    xs = x.cpu().numpy()[keep]
    imgs = snaps.cpu().numpy()[:, :, keep][:, :, :, keep]  # (F, B, n, n)
    st = snap_times.cpu().numpy()
    stride = max(1, len(times) // 1500)
    t, rec = times[::stride].cpu().numpy(), f[::stride].cpu().numpy()  # (T,), (T, B)
    t_fine = torch.linspace(0, float(times[-1]), 400, dtype=times.dtype)
    fitted = soumpasis(t_fine, w**2 / (4 * D_fit.cpu())).numpy()  # (400, B)
    t_f, th = t_fine.numpy(), t_half.cpu().numpy()
    B = len(names)
    colors = [f"C{b}" for b in range(B)]

    fig = plt.figure(figsize=(8.0, 5.0))
    grid = fig.add_gridspec(2, B, height_ratios=[1, 1.05])
    cmap = plt.get_cmap("Greens", 10)  # discrete levels: small GIF, contours easy to read
    extent = (xs[0], xs[-1], xs[0], xs[-1])
    ims = []
    circle = np.linspace(0, 2 * np.pi, 100)
    for b in range(B):
        ax = fig.add_subplot(grid[0, b])
        ims.append(ax.imshow(imgs[0, b].T, origin="lower", extent=extent, cmap=cmap, vmin=depth, vmax=1.0))
        ax.plot(w * np.cos(circle), w * np.sin(circle), "--", color="#f59e0b", lw=1)  # bleached disk
        ax.set_title(f"{names[b]}\nD = {D[b]:g} µm²/s", fontsize=9, color=colors[b])
        ax.set_xticks([-ZOOM * w, 0, ZOOM * w])
        ax.set_yticks([-ZOOM * w, 0, ZOOM * w] if b == 0 else [])
        ax.grid(False)
        if b == 0:
            ax.set_ylabel("y (µm)")
        ax.set_xlabel("x (µm)", labelpad=1)
    fig.colorbar(ims[-1], ax=fig.axes[:B], fraction=0.03, pad=0.02, label="fluorescence")

    ax_r = fig.add_subplot(grid[1, :])
    ax_r.set_xlim(0, t[-1])
    ax_r.set_ylim(0, 1.08)
    ax_r.set_xlabel("time after bleach (s)")
    ax_r.set_ylabel("recovery f(t)")
    ax_r.axhline(0.5, color="#9ca3af", ls="--", lw=1)
    for b in range(B):
        ax_r.plot(t_f, fitted[:, b], "--", lw=1, color="#f59e0b")
        ax_r.plot([th[b]], [0.5], "|", ms=10, mew=1.5, color=colors[b])
    curves = [
        ax_r.plot([], [], lw=2, color=colors[b], label=f"{names[b]}: t½ = {th[b]:.2f} s, D fit = {D_fit[b]:.3g}")[0]
        for b in range(B)
    ]
    ax_r.plot([], [], "--", lw=1, color="#f59e0b", label="Soumpasis fit")
    ax_r.legend(loc="lower right", fontsize=8)
    clock = ax_r.text(0.01, 0.92, "", transform=ax_r.transAxes, fontsize=9)

    def update(k: int):
        for b in range(B):
            ims[b].set_data(imgs[k, b].T)
            n = np.searchsorted(t, st[k], side="right")
            curves[b].set_data(t[:n], rec[:n, b])
        clock.set_text(f"t = {st[k]:.1f} s")

    fig.subplots_adjust(left=0.08, right=0.88, top=0.9, bottom=0.1, hspace=0.45, wspace=0.12)
    return save_gif(fig, update, list(range(len(st))), path)
