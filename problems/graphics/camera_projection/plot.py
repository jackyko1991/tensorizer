"""Three cameras side by side: perspective (with vanishing points), orthographic, and a dolly zoom."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from problems._common.plotting import save_gif


def segments(px: np.ndarray, ok: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Line segments (E', 2, 2) for the edges whose two ends are in front of the camera."""
    keep = ok[edges[:, 0]] & ok[edges[:, 1]]
    return px[edges[keep]]


def animate(res: dict, edges: torch.Tensor, grid_edges: torch.Tensor, w: int, h: int, fov: float, path: Path) -> Path:
    r = {k: v.cpu().numpy() for k, v in res.items()}
    e, ge = edges.cpu().numpy(), grid_edges.cpu().numpy()
    n_lines = len(ge) // 2  # first half of the grid edges run along z, second half along x

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    titles = ["perspective", "orthographic", "dolly zoom"]
    grids, objs = [], []
    for c, (ax, title) in enumerate(zip(axes, titles)):
        frame = ax.add_patch(Rectangle((0, 0), w, h, facecolor="#f9fafb", edgecolor="#374151", lw=1.2, zorder=0))
        grids.append(ax.add_collection(LineCollection([], colors="#9ca3af", lw=0.8, zorder=2)))
        objs.append(ax.add_collection(LineCollection([], colors="#2563eb", lw=1.6, zorder=3)))
        if c == 0:  # room around the image for the vanishing points
            ax.set_xlim(-0.6 * w, 1.6 * w)
            ax.set_ylim(1.2 * h, -0.45 * h)  # image rows grow downwards
        else:  # just the image: clip the lines at its border
            grids[c].set_clip_path(frame)
            objs[c].set_clip_path(frame)
            ax.set_xlim(-0.03 * w, 1.03 * w)
            ax.set_ylim(1.03 * h, -0.03 * h)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(title, fontsize=10)
    guides = axes[0].add_collection(LineCollection([], colors="#dc2626", lw=0.6, linestyles="--", alpha=0.5, zorder=1))
    (vps,) = axes[0].plot([], [], "o", color="#dc2626", ms=6, label="vanishing points")
    (horizon,) = axes[0].plot([], [], "--", color="#f59e0b", lw=1.2, label="horizon")
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.9)
    axes[1].text(0.5, 0.04, "parallel lines stay parallel", transform=axes[1].transAxes, ha="center", fontsize=8)
    dolly_label = axes[2].text(0.5, 0.04, "", transform=axes[2].transAxes, ha="center", fontsize=8)
    axes[0].set_title(f"perspective, fov {fov:.0f}°", fontsize=10)

    def update(f: int):
        for c in range(3):
            grids[c].set_segments(segments(r["grid"][f, c], r["grid_ok"][f, c], ge))
            objs[c].set_segments(segments(r["obj"][f, c], r["obj_ok"][f, c], e))
        # extend every floor line from its end nearer the vanishing point of its family (z-lines → vp[1], x → vp[0])
        px, vp = r["grid"][f, 0], r["vp"][f]
        lines = []
        for j, (a, b) in enumerate(ge):
            target = vp[1] if j < n_lines else vp[0]
            if np.all(np.isfinite(target)):
                near = px[a] if np.hypot(*(px[a] - target)) < np.hypot(*(px[b] - target)) else px[b]
                lines.append([near, target])
        guides.set_segments(lines)
        vps.set_data(vp[:, 0], vp[:, 1])
        horizon.set_data([-w, 2 * w], [r["horizon"][f]] * 2)
        dolly_label.set_text(f"fov {r['dolly_fov'][f]:.0f}°, distance {r['dolly_dist'][f]:.1f}")

    fig.tight_layout()
    return save_gif(fig, update, list(range(len(r["vp"]))), path, fps=12)
