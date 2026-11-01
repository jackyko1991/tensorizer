"""Animate the rank-k reconstruction growing layer by layer, with the spectrum, the first layers and PSNR vs storage."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import layer, reconstruct


def _image_axis(ax, title: str):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=9)


def animate(rgb: torch.Tensor, grey_svd: tuple, rgb_svd: tuple, curve: tuple, ranks: list[int], path: Path) -> Path:
    """rgb (3, H, W); grey_svd = (U, σ, V) of the grey image; rgb_svd the batched version; curve = (k, PSNR, ratio)."""
    u, s, v = grey_svd
    u_c, s_c, v_c = rgb_svd
    a = rgb.permute(1, 2, 0).cpu().numpy()  # (H, W, 3) for imshow
    grey = reconstruct(u, s, v, len(s))
    _, psnr, ratio = (t.cpu().numpy() for t in curve)
    s_np, s_c_np = s.cpu().numpy(), s_c.cpu().numpy()

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.9, 0.9])
    ax_a, ax_k, ax_e = (fig.add_subplot(gs[0, j]) for j in range(3))
    ax_s = fig.add_subplot(gs[0, 3:])
    ax_l = [fig.add_subplot(gs[1, j]) for j in range(3)]
    ax_q = fig.add_subplot(gs[1, 3:])

    ax_a.imshow(a, interpolation="nearest")
    _image_axis(ax_a, "original $A_{cij}$")
    img_k = ax_k.imshow(a, interpolation="nearest")
    _image_axis(ax_k, "")
    img_e = ax_e.imshow(np.zeros(a.shape[:2]), cmap="Greys", vmin=0, vmax=0.5, interpolation="nearest")
    _image_axis(ax_e, "grey error $|A - A_k|$")

    for r, ax in enumerate(ax_l):  # the first rank-1 layers σ_r u_r v_rᵀ of the grey image
        lay = layer(u, s, v, r).cpu().numpy()
        lim = np.abs(lay).max()
        ax.imshow(lay, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        _image_axis(ax, rf"layer {r + 1}: $\sigma_{r + 1} u_{r + 1} v_{r + 1}^T$, $\sigma$ = {s_np[r]:.1f}")

    rr = np.arange(1, len(s_np) + 1)
    for c, col in enumerate(["#dc2626", "#16a34a", "#2563eb"]):
        ax_s.semilogy(rr, s_c_np[c], color=col, lw=0.8, alpha=0.7)
    ax_s.semilogy(rr, s_np, color="#111827", lw=1.4, label="grey")
    tail = ax_s.axvspan(1, 1, color="#f59e0b", alpha=0.15, label=r"dropped: $\sum_{r>k}\sigma_r^2$")
    ax_s.set_xlabel("index $r$")
    ax_s.set_ylabel(r"$\sigma_r$")
    ax_s.set_title("singular values (R, G, B and grey)", fontsize=9)
    ax_s.legend(fontsize=7, loc="upper right")

    ax_q.plot(ratio, psnr, color="#2563eb", lw=1.4)
    ax_q.set_xscale("log")
    ax_q.set_xlabel("compression ratio")
    ax_q.set_ylabel("PSNR (dB)")
    ax_q.set_ylim(10, min(60, psnr.max() + 3))
    ax_q.set_title("quality vs storage (colour)", fontsize=9)
    (dot,) = ax_q.plot([], [], "o", color="#dc2626", ms=7)

    def update(n: int):
        k = ranks[n]
        a_k = reconstruct(u_c, s_c, v_c, k).clamp(0, 1)  # all three channels in one batched einsum
        img_k.set_data(a_k.permute(1, 2, 0).cpu().numpy())
        ax_k.set_title(f"rank $k$ = {k}: {ratio[k - 1]:.1f}× smaller", fontsize=9)
        img_e.set_data((grey - reconstruct(u, s, v, k)).abs().cpu().numpy())
        tail.set_x(k + 0.5)
        tail.set_width(len(s_np) - k)
        dot.set_data([ratio[k - 1]], [psnr[k - 1]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(ranks), 120), path, fps=4)
