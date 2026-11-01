"""Progressive render (GI and direct only), the indirect light of the reference, and the noise against spp."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def tonemap(img: torch.Tensor) -> np.ndarray:
    """Linear radiance → displayable sRGB-like values (clip, then gamma 2.2)."""
    return img.clamp(0, 1).pow(1 / 2.2).cpu().numpy()


def animate(gi, di, ref_gi, ref_di, milestones, err_gi, err_di, ref_spp, path: Path) -> Path:
    """gi, di (M, H, W, 3) running means at the milestone sample counts; ref_* (H, W, 3)."""
    ns = np.array(milestones)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6))
    (ax_gi, ax_di), (ax_ind, ax_err) = axes
    im_gi = ax_gi.imshow(tonemap(gi[0]), interpolation="nearest")
    im_di = ax_di.imshow(tonemap(di[0]), interpolation="nearest")
    indirect = (ref_gi - ref_di).clamp(min=0)
    ax_ind.imshow(tonemap(indirect * 3), interpolation="nearest")
    ax_ind.set_title(f"indirect light only (×3), {ref_spp} spp", fontsize=10)
    for ax in (ax_gi, ax_di, ax_ind):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    ax_err.loglog(ns, err_gi.numpy(), color="#2563eb", lw=1.5, label="global illumination")
    ax_err.loglog(ns, err_di.numpy(), color="#ea580c", lw=1.5, label="direct only")
    ax_err.loglog(ns, err_gi[0].item() / np.sqrt(ns), "--", color="#9ca3af", lw=1.2, label=r"$\propto 1/\sqrt{spp}$")
    (dot,) = ax_err.plot([], [], "o", color="#2563eb", ms=6)
    ax_err.set_xlabel("samples per pixel")
    ax_err.set_ylabel("RMSE vs reference")
    ax_err.set_title("noise", fontsize=10)
    ax_err.legend(fontsize=8, loc="lower left")

    def update(m: int):
        im_gi.set_data(tonemap(gi[m]))
        im_di.set_data(tonemap(di[m]))
        ax_gi.set_title(f"path traced (GI), {ns[m]} spp", fontsize=10)
        ax_di.set_title(f"direct light only, {ns[m]} spp", fontsize=10)
        dot.set_data([ns[m]], [err_gi[m].item()])

    update(0)
    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(ns), 120), path, fps=6)
