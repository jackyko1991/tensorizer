"""Animate the field: direction of B in the x–z plane, and B_z on the axis against the exact formula."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif

from .tensors import EXTENT, LAYOUTS

TESLA_TO_MICRO = 1e6


def animate(result: dict, layout: str, path: Path) -> Path:
    p = result["plane"].cpu().numpy()  # (G, 3)
    B = result["B_plane"].cpu().numpy()  # (T, G, 3)
    strength = np.linalg.norm(B, axis=-1)  # (T, G)
    Bx, Bz = B[..., 0] / strength, B[..., 2] / strength  # unit arrows: only the direction is drawn
    z = result["axis"][:, 2].cpu().numpy()  # (A,)
    Bz_axis = result["B_axis"][..., 2].cpu().numpy() * TESLA_TO_MICRO  # (T, A)
    Bz_exact = result["B_axis_exact"].cpu().numpy() * TESLA_TO_MICRO  # (T, A)
    radii = result["radii"].numpy()  # (T,)
    log_B = np.log10(strength * TESLA_TO_MICRO)
    lo, hi = np.quantile(log_B, [0.02, 0.98])

    fig, (ax, ax_axis) = plt.subplots(1, 2, figsize=(7, 4))

    # left: arrows coloured by log |B|; where each loop pierces the plane: ⊙ current out of the page, ⊗ into it
    arrows = ax.quiver(
        p[:, 0], p[:, 2], Bx[0], Bz[0], log_B[0], cmap="viridis", clim=(lo, hi), pivot="mid", scale=32, width=0.006
    )
    crossings = []
    for height, sign in LAYOUTS[layout]:
        for side in (-1, +1):
            symbol = "⊗" if side * sign > 0 else "⊙"
            crossings.append((side, height, ax.text(0, 0, symbol, ha="center", va="center", fontsize=14)))
    ax.axvline(0, color="#9ca3af", lw=0.8, ls=":")
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    title = ax.set_title("")

    # right: field along the dotted axis x = 0 — the summed segments against the textbook formula
    (exact_line,) = ax_axis.plot([], [], color="k", lw=1.5, label="exact")
    (numeric_line,) = ax_axis.plot([], [], "o", ms=3, color="C1", label="Σ over segments")
    ax_axis.set_xlim(-EXTENT, EXTENT)
    ax_axis.set_ylim(min(0, Bz_exact.min()) * 1.1, Bz_exact.max() * 1.1)
    ax_axis.set_xlabel("z on the axis (m)")
    ax_axis.set_ylabel("B_z (μT)")
    ax_axis.legend(loc="upper right", fontsize=8)
    ax_axis.set_title("field on the axis")

    every = 6  # draw every 6th numerical point so the exact curve stays visible

    def update(step: int):
        arrows.set_UVC(Bx[step], Bz[step], log_B[step])
        for side, height, label in crossings:
            label.set_position((side * radii[step], height * radii[step]))
        exact_line.set_data(z, Bz_exact[step])
        numeric_line.set_data(z[::every], Bz_axis[step, ::every])
        title.set_text(f"direction of B (colour: log₁₀|B|),  R = {radii[step]:.2f} m")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(radii), 72), path, fps=12)
