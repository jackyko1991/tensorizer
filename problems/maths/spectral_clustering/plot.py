"""Animate the points sliding into their spectral embedding (top), k-means for comparison (bottom),
the spectra and the vibration modes of a grid graph."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

N_FRAMES = 60


def morph_weight(f: int) -> float:
    """0 = data space, 1 = spectral embedding: hold, slide over, hold, slide back."""
    t = (f % N_FRAMES) / N_FRAMES
    ramp = np.clip((t - 0.1) / 0.3, 0, 1) if t < 0.55 else np.clip(1 - (t - 0.65) / 0.3, 0, 1)
    return float(3 * ramp**2 - 2 * ramp**3)


def to_box(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rescale the first two embedding coordinates into the bounding box of the data, so both fit one axis."""
    u = u[:, :2]
    lo, hi = u.min(0), u.max(0)
    return x.min(0) + (u - lo) / np.maximum(hi - lo, 1e-9) * (x.max(0) - x.min(0))


def animate(results: dict, modes: torch.Tensor, mode_evals: torch.Tensor, g: int, path: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.2))
    colours = np.array(ACCENT[:3])
    moving = {}
    for col, (name, r) in enumerate(results.items()):
        x = r["x"].cpu().numpy()
        target = to_box(r["embedding"].cpu().numpy(), x)
        ax_s, ax_k = axes[0, col], axes[1, col]
        dots = ax_s.scatter(x[:, 0], x[:, 1], s=7, c=colours[r["spectral"].cpu().numpy()])
        moving[name] = (dots, x, target)
        ax_s.set_title(f"{name}: spectral, ARI {r['ari'][0]:.2f}", fontsize=9)
        ax_k.scatter(x[:, 0], x[:, 1], s=7, c=colours[r["kmeans"].cpu().numpy()])
        ax_k.set_title(f"{name}: plain k-means, ARI {r['ari'][1]:.2f}", fontsize=9)
        for ax in (ax_s, ax_k):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
    label = fig.text(0.37, 0.965, "", ha="center", fontsize=10, color="#374151")

    ax_f = axes[0, 3]  # Fiedler vector of the moons graph, with the spectra as an inset
    moons = results["moons"]
    xm, fm = moons["x"].cpu().numpy(), moons["fiedler"].cpu().numpy()
    ax_f.scatter(xm[:, 0], xm[:, 1], s=7, c=fm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_f.set_title("moons: Fiedler vector $f_2$ (sign = cut)", fontsize=9)
    ax_f.set_xticks([])
    ax_f.set_yticks([])
    ax_e = ax_f.inset_axes((0.6, 0.6, 0.37, 0.28))
    for col, r in enumerate(results.values()):
        ev = r["evals"].cpu().numpy()
        ax_e.plot(np.arange(1, len(ev) + 1), ev, "o-", ms=2, lw=0.8, color=ACCENT[col])
    ax_e.set_title(r"$\lambda_m(L_{sym})$, 3 sets", fontsize=7, pad=2)
    ax_e.tick_params(labelsize=5, length=2)

    ax_g = axes[1, 3]
    shapes = modes.T.reshape(-1, g, g).cpu().numpy()  # (M, g, g)
    vmax = np.abs(shapes).max()
    img = ax_g.imshow(shapes[0], cmap=plt.get_cmap("RdBu_r", 15), vmin=-vmax, vmax=vmax)
    ax_g.set_xticks([])
    ax_g.set_yticks([])
    ax_g.grid(False)
    title_g = ax_g.set_title("", fontsize=9)
    per_mode = N_FRAMES / len(shapes)

    def update(f: int):
        s = morph_weight(f)
        for dots, x, target in moving.values():
            dots.set_offsets((1 - s) * x + s * target)
        label.set_text(f"top: data space → spectral embedding (weight {s:.2f}), coloured by spectral clusters")
        m = min(int(f / per_mode), len(shapes) - 1)
        phase = (f - m * per_mode) / per_mode
        img.set_data(shapes[m] * np.cos(2 * np.pi * phase))
        title_g.set_text(f"grid mode {m + 2}, $\\lambda$ = {mode_evals[m]:.3f}")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_gif(fig, update, frame_indices(N_FRAMES, N_FRAMES), path, fps=10)
