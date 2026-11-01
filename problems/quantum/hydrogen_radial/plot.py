"""Cycle through the orbitals: radial probability u² (left) and a slice through |ψ_nl0|² (right)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import PowerNorm

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(r: torch.Tensor, orbitals: list[dict], path: Path, hold: int = 15, draw_in: int = 6) -> Path:
    """Each orbital is shown for `hold` frames; its radial curve is drawn in over the first `draw_in`."""
    rs = r.cpu().numpy()
    n_max = max(o["n"] for o in orbitals)
    r_view = 3.0 * n_max**2 + 5.0

    fig, (ax, ax_img) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax.set_xlim(0, r_view)
    ax.set_xlabel("r  (Bohr radii)")
    ax.set_ylabel("radial probability  r²|R|² = u²")
    (line,) = ax.plot([], [], color=ACCENT[0], lw=2)
    fill = [ax.fill_between(rs, 0 * rs, color=ACCENT[0], alpha=0.2)]
    label = ax.text(0.97, 0.95, "", transform=ax.transAxes, va="top", ha="right", family="monospace", fontsize=9)

    first = orbitals[0]
    image = ax_img.imshow(first["density"].cpu().numpy(), origin="lower", cmap="magma", interpolation="bilinear")
    ax_img.set_xlabel("x")
    ax_img.set_ylabel("z")
    ax_img.grid(False)
    title = ax_img.set_title("")

    def update(step: int):
        orbital = orbitals[min(step // hold, len(orbitals) - 1)]
        P = orbital["P"].cpu().numpy()
        shown = min(1.0, (step % hold + 1) / draw_in)  # fraction of the curve drawn so far
        n_drawn = max(2, int(shown * (rs < r_view).sum()))
        line.set_data(rs[:n_drawn], P[:n_drawn])
        fill[0].remove()
        fill[0] = ax.fill_between(rs[:n_drawn], P[:n_drawn], color=ACCENT[0], alpha=0.2)
        ax.set_ylim(0, 1.15 * P.max())
        n, ell, E = orbital["n"], orbital["l"], orbital["E"]
        label.set_text(
            f"{orbital['name']}  (n={n}, l={ell})\nE = {E:.4f}\n-1/2n² = {-0.5 / n**2:.4f}\nnodes = {orbital['nodes']}"
        )

        density = orbital["density"].cpu().numpy()
        a = orbital["half"]
        image.set_data(density)
        image.set_extent((-a, a, -a, a))
        image.set_norm(PowerNorm(0.3, vmin=0, vmax=density.max()))  # compressed colour scale shows faint lobes
        title.set_text(f"|ψ_{n}{ell}0|² in the xz-plane")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(hold * len(orbitals)), path, fps=10)
