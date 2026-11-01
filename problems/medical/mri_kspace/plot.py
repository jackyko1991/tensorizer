"""Animate k-space filling with the sharpening reconstruction, then compare three undersampling schemes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def _kspace_view(s: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """log10 |s| with the zero frequency in the centre; unmeasured samples masked out (drawn grey)."""
    view = np.fft.fftshift(np.log10(np.abs(s) + 1e-2))
    return np.ma.masked_where(np.fft.fftshift(mask) == 0, view)


def animate(
    s: torch.Tensor,
    rho: torch.Tensor,
    masks: torch.Tensor,
    images: torch.Tensor,
    results: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    path: Path,
) -> Path:
    s_np, truth = s.cpu().numpy(), rho.cpu().numpy()
    masks_np, images_np = masks.cpu().numpy(), images.cpu().numpy()
    n_fill = len(masks_np)
    hold = max(1, n_fill // 4)  # frames spent on each undersampling scheme
    total = n_fill + hold * len(results)

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.2))
    cmap_k = plt.get_cmap("viridis").with_extremes(bad="#e5e7eb")  # unmeasured samples in light grey
    full = np.fft.fftshift(np.log10(np.abs(s_np) + 1e-2))
    k_im = axes[0].imshow(_kspace_view(s_np, masks_np[0]), cmap=cmap_k, vmin=full.min(), vmax=full.max())
    rec_im = axes[1].imshow(images_np[0], cmap="gray", vmin=0, vmax=0.6)
    ref_im = axes[2].imshow(truth, cmap="gray", vmin=0, vmax=0.6)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    titles = [ax.set_title("", fontsize=9) for ax in axes]
    banner = fig.suptitle("", fontsize=10)

    def update(step: int):
        if step < n_fill:
            lines = int(masks_np[step][:, 0].sum())
            k_im.set_data(_kspace_view(s_np, masks_np[step]))
            rec_im.set_data(images_np[step])
            ref_im.set_data(truth)
            banner.set_text(f"Cartesian filling, centre out: {lines}/{len(truth)} phase-encode lines")
            titles[0].set_text("k-space  log|s(k)|")
            titles[1].set_text("inverse FFT of lines so far")
            titles[2].set_text("phantom ρ(x)")
            return
        name, mask, zf, reg = results[min((step - n_fill) // hold, len(results) - 1)]
        k_im.set_data(_kspace_view(s_np, mask.cpu().numpy()))
        rec_im.set_data(zf.cpu().numpy())
        ref_im.set_data(reg.cpu().numpy())
        banner.set_text(f"Undersampled: {name}")
        titles[0].set_text("acquired k-space")
        titles[1].set_text("zero-filled IFFT")
        titles[2].set_text("regularised least squares")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(total, total), path, fps=10)
