"""Animate the zoom into M next to a Julia set whose c walks around the main cardioid, plus an overview map."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from problems._common.plotting import save_gif

# white far away → blue near the set; the set itself and its thin filaments are dark (the "bad" colour)
COLOURS = ["#ffffff", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]
CMAP = LinearSegmentedColormap.from_list("escape", COLOURS, N=16).with_extremes(bad="#111827")


def shade(result: dict, pixel: float) -> np.ma.MaskedArray:
    """log ν scaled to [0, 1] per frame; interior pixels and those within ¼ pixel of the boundary are masked."""
    nu, d = result["nu"].cpu().numpy(), result["d"].cpu().numpy()
    value = np.log(np.where(np.isnan(nu), 1.0, np.maximum(nu, 1.0)))
    lo, hi = np.nanpercentile(np.where(np.isnan(nu), np.nan, value), [1, 99])
    value = np.clip((value - lo) / max(hi - lo, 1e-9), 0, 1)
    return np.ma.masked_where(np.isnan(nu) | (d < 0.25 * pixel), value)


def animate(
    zoom: list[dict],
    widths: list[float],
    centres: list[complex],
    julia: list[dict],
    c_julia: torch.Tensor,
    overview: dict,
    overview_box: tuple[float, float, float, float],
    path: Path,
) -> Path:
    """zoom/julia: one result dict per frame with (n, n) tensors; overview: a result dict for the whole set."""
    n = zoom[0]["nu"].shape[-1]
    fig, (ax_z, ax_j, ax_o) = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax in (ax_z, ax_j, ax_o):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    img_z = ax_z.imshow(shade(zoom[0], widths[0] / n), cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    img_j = ax_j.imshow(shade(julia[0], 3.2 / n), cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    title_z = ax_z.set_title("", fontsize=9)
    title_j = ax_j.set_title("", fontsize=9)

    x0, x1, y0, y1 = overview_box
    ax_o.imshow(shade(overview, (x1 - x0) / overview["nu"].shape[-1]), cmap=CMAP, extent=overview_box, vmin=0, vmax=1)
    theta = np.linspace(0, 2 * np.pi, 400)
    w = np.exp(1j * theta)
    edge = w / 2 - w * w / 4
    ax_o.plot(edge.real, edge.imag, color="#f59e0b", lw=1.2, label="cardioid $|2z^*| = 1$")
    ax_o.plot(-1 + 0.25 * np.cos(theta), 0.25 * np.sin(theta), color="#16a34a", lw=1.2, label="bulb $|c+1| = 1/4$")
    (dot_c,) = ax_o.plot([], [], "o", color="#dc2626", ms=6, label="Julia $c$")
    box = plt.Rectangle((0, 0), 0, 0, fill=False, ec="#dc2626", lw=1.2, ls="--")
    ax_o.add_patch(box)
    ax_o.set_xlim(x0, x1)
    ax_o.set_ylim(y0, y1)
    ax_o.set_title("overview: exact tests vs iteration", fontsize=9)
    ax_o.legend(loc="lower left", fontsize=7, framealpha=0.9)

    c_np = c_julia.cpu().numpy()

    def update(f: int):
        img_z.set_data(shade(zoom[f], widths[f] / n))
        img_j.set_data(shade(julia[f], 3.2 / n))
        c, w_f = centres[f], widths[f]
        title_z.set_text(f"zoom ×{3.2 / w_f:.1e} at {c.real:+.6f} {c.imag:+.6f}i")
        title_j.set_text(f"Julia set, c = {c_np[f].real:+.3f} {c_np[f].imag:+.3f}i")
        dot_c.set_data([c_np[f].real], [c_np[f].imag])
        size = max(w_f, 0.04)  # keep the box visible once it is smaller than a pixel of the overview
        box.set_bounds(c.real - size / 2, c.imag - size / 2, size, size)

    fig.tight_layout()
    return save_gif(fig, update, list(range(len(zoom))), path, fps=8)
