"""Four famous rules growing row by row, a mosaic of all 256 rules, and the Lenia glider with its mass."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from problems._common.plotting import frame_indices, save_gif

CMAP = ListedColormap(["#ffffff", "#1e3a8a"])
KIND_RGB = np.array([[156, 163, 175], [37, 99, 235], [220, 38, 38]]) / 255  # uniform, periodic, no period
LABEL = {30: "chaos", 90: "Sierpinski", 110: "complex", 184: "traffic"}


def mosaic_image(mosaic: torch.Tensor, kind: torch.Tensor) -> np.ndarray:
    """(T+1, 256, W) diagrams → one 16 × 16 RGB tile image, live cells coloured by behaviour class."""
    t, _, w = mosaic.shape
    tiles = mosaic.permute(1, 0, 2).cpu().numpy().reshape(16, 16, t, w)  # rule r = 16·row + column
    pad = np.zeros((16, 16, t + 2, w + 2))
    pad[:, :, 1:-1, 1:-1] = tiles
    img = np.ones((16 * (t + 2), 16 * (w + 2), 3))
    live = pad.transpose(0, 2, 1, 3).reshape(16 * (t + 2), 16 * (w + 2)) > 0
    colours = KIND_RGB[kind.cpu().numpy()].reshape(16, 16, 3)
    rgb = np.broadcast_to(colours[:, None, :, None, :], (16, t + 2, 16, w + 2, 3)).reshape(img.shape)
    img[live] = rgb[live]
    return img


def animate(
    spacetime: torch.Tensor,
    rules: list[int],
    mosaic: torch.Tensor,
    kind: torch.Tensor,
    frames: torch.Tensor,
    mass: torch.Tensor,
    path: Path,
) -> Path:
    """spacetime (T+1, 4, W), mosaic (T'+1, 256, W'), kind (256,), frames (L+1, S, S), mass (L+1,)."""
    st = spacetime.cpu().numpy().astype(float)
    n_rows = st.shape[0]
    lenia = frames.cpu().numpy()
    m = mass.cpu().numpy()

    fig = plt.figure(figsize=(10, 6.4))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.35, wspace=0.25)
    images = []
    for j, r in enumerate(rules):
        ax = fig.add_subplot(grid[0, j])
        im = ax.imshow(
            np.full_like(st[:, j], np.nan), cmap=CMAP, vmin=0, vmax=1, interpolation="nearest", aspect="auto"
        )
        ax.set_title(f"rule {r}" + (f" ({LABEL[r]})" if r in LABEL else ""), fontsize=10)
        ax.set_xticks([])
        ax.set_ylabel("time ↓" if j == 0 else "")
        ax.set_yticks([])
        images.append(im)

    ax_m = fig.add_subplot(grid[1, :2])
    ax_m.imshow(mosaic_image(mosaic, kind), interpolation="nearest")
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    ax_m.set_title("all 256 rules (row-major, rule 0 top left)", fontsize=10)
    for c, name in enumerate(["uniform", "periodic", "no period found"]):
        ax_m.plot([], [], "s", color=KIND_RGB[c], label=name)
    ax_m.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)

    ax_l = fig.add_subplot(grid[1, 2])
    lenia_im = ax_l.imshow(lenia[0], cmap="magma", vmin=0, vmax=1, interpolation="bilinear")
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    lenia_title = ax_l.set_title("", fontsize=10)

    ax_c = fig.add_subplot(grid[1, 3])
    steps = np.arange(len(m))
    ax_c.plot(steps, m, color="#9ca3af", lw=1)
    (trace,) = ax_c.plot([], [], color="#dc2626", lw=1.6)
    ax_c.set_ylim(0, 1.3 * m.max())
    ax_c.set_xlabel("Lenia step")
    ax_c.set_title("mass Σ A (soliton: flat)", fontsize=10)

    def update(t: int):
        for j, im in enumerate(images):
            shown = st[:, j].copy()
            shown[t + 1 :] = np.nan  # rows not grown yet
            im.set_data(shown)
        s = round(t * (len(m) - 1) / max(1, n_rows - 1))
        lenia_im.set_data(lenia[s])
        lenia_title.set_text(f"Lenia Orbium, step {s}")
        trace.set_data(steps[: s + 1], m[: s + 1])

    return save_gif(fig, update, frame_indices(n_rows, 90), path, fps=10)
