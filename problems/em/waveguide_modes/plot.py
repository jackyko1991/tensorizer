"""Animate the modes one after another: the field pattern oscillating in time, and the list of cutoffs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

FRAMES_PER_MODE = 16


def animate(found: list[dict], frequencies: list[float], a: float, b: float, path: Path) -> Path:
    patterns = [m["psi"].numpy() / np.abs(m["psi"].numpy()).max() for m in found]  # each scaled to ±1
    arrows = [m["E_t"].numpy() for m in found]  # (2, Nx, Ny) each
    nx, ny = patterns[0].shape
    every = max(1, ny // 8)  # one arrow every few cells
    x = (np.arange(nx) + 0.5) * a / nx
    y = (np.arange(ny) + 0.5) * b / ny
    X, Y = np.meshgrid(x[::every], y[::every], indexing="ij")
    n_frames = len(found) * FRAMES_PER_MODE

    fig, (ax, ax_list) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    image = ax.imshow(
        patterns[0].T, origin="lower", extent=(0, a, 0, b), cmap=plt.get_cmap("RdBu_r", 16), vmin=-1, vmax=1
    )
    quiver = ax.quiver(X, Y, X * 0, Y * 0, color="#111827", pivot="mid", scale=None)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.grid(False)
    title = ax.set_title("")

    # right: every mode with its cutoff; the one on screen is highlighted
    ax_list.axis("off")
    ax_list.set_title("cutoff frequencies", fontsize=10)
    rows = []
    for k, (mode, f) in enumerate(zip(found, frequencies)):
        m, n = mode["mn"]
        text = f"{mode['kind']}$_{{{m}{n}}}$   {f:5.2f} GHz"
        rows.append(ax_list.text(0.05, 0.92 - k * 0.08, text, transform=ax_list.transAxes, fontsize=10))
    ax_list.text(0.05, 0.92 - len(found) * 0.08 - 0.04, "arrows: transverse E", fontsize=8,
                 transform=ax_list.transAxes, color="#6b7280")  # fmt: skip

    def update(frame: int):
        k, phase = divmod(frame, FRAMES_PER_MODE)
        swing = np.cos(2 * np.pi * phase / FRAMES_PER_MODE)  # the mode oscillates as cos(ωt)
        image.set_data(swing * patterns[k].T)
        E = arrows[k][:, ::every, ::every]
        scale = np.abs(E).max() + 1e-12
        quiver.set_UVC(swing * E[0] / scale, swing * E[1] / scale)
        mode = found[k]
        field = "E_z" if mode["kind"] == "TM" else "H_z"
        m, n = mode["mn"]
        title.set_text(f"{mode['kind']}$_{{{m}{n}}}$: colour = {field},  k_c = {mode['k_c']:.3f} /cm")
        for j, row in enumerate(rows):
            row.set_color(ACCENT[1] if j == k else "#374151")
            row.set_fontweight("bold" if j == k else "normal")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(n_frames, n_frames), path, fps=10)
