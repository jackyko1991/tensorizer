"""Animate training: held-out view and depth map with and without positional encoding, PSNR, and ray weights."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import save_gif

from .tensors import FAR, NEAR

BLUE, RED, GREY, DARK = "#2563eb", "#dc2626", "#9ca3af", "#374151"


def _img(x: torch.Tensor, res: int) -> np.ndarray:
    """(P, 3) colours → (res, res, 3) image; (P,) depths → (res, res) map."""
    return x.reshape(res, res, 3).clamp(0, 1).numpy() if x.dim() == 2 else x.reshape(res, res).numpy()


def animate(rgb_true: torch.Tensor, depth_true: torch.Tensor, runs: dict, res: int, n_freq: int, path: Path) -> Path:
    pe, raw = runs["with encoding"]["snaps"], runs["without encoding"]["snaps"]
    its = np.array([s["it"] for s in pe])
    p_pe, p_raw = np.array([s["psnr"] for s in pe]), np.array([s["psnr"] for s in raw])
    q_pe, q_raw = np.array([s["psnr_train"] for s in pe]), np.array([s["psnr_train"] for s in raw])
    d_lo, d_hi = float(depth_true.min()) - 0.1, FAR
    centre = res * res // 2 + res // 2
    t_surface = float(depth_true[centre])

    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.8))
    titles = ["held-out view: truth", f"NeRF, encoding L = {n_freq}", "NeRF, no encoding",
              "true depth", "depth $\\sum_s w_s t_s$", "depth, no encoding"]  # fmt: skip
    for ax, title in zip([*axes[:, 0], *axes[:, 1], *axes[:, 2]], [titles[i] for i in (0, 3, 1, 4, 2, 5)]):
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0, 0].imshow(_img(rgb_true, res))
    axes[1, 0].imshow(_img(depth_true, res), cmap="viridis_r", vmin=d_lo, vmax=d_hi)
    im_pe = axes[0, 1].imshow(_img(pe[0]["rgb"], res))
    im_raw = axes[0, 2].imshow(_img(raw[0]["rgb"], res))
    dm_pe = axes[1, 1].imshow(_img(pe[0]["depth"], res), cmap="viridis_r", vmin=d_lo, vmax=d_hi)
    dm_raw = axes[1, 2].imshow(_img(raw[0]["depth"], res), cmap="viridis_r", vmin=d_lo, vmax=d_hi)
    axes[1, 0].plot([res // 2], [res // 2], "+", color=RED, ms=7)  # the ray shown bottom right

    ax_p = axes[0, 3]
    (l_pe,) = ax_p.plot([], [], "o-", color=BLUE, ms=2.5, lw=1.2, label=f"encoding L = {n_freq}")
    (l_raw,) = ax_p.plot([], [], "s-", color=RED, ms=2.5, lw=1.2, label="no encoding")
    (k_pe,) = ax_p.plot([], [], "--", color=BLUE, lw=1, label="training view")
    (k_raw,) = ax_p.plot([], [], "--", color=RED, lw=1)
    ax_p.set_xlim(0, its[-1])
    ax_p.set_ylim(min(p_pe.min(), p_raw.min()) - 1, max(q_pe.max(), q_raw.max(), p_pe.max()) + 3)
    ax_p.set_title("PSNR (dB): held-out, - - training", fontsize=8.5)
    ax_p.set_xlabel("Adam step", fontsize=7.5)
    ax_p.tick_params(labelsize=7)
    ax_p.legend(fontsize=6, loc="lower right")

    ax_w = axes[1, 3]
    t = pe[0]["t"].numpy()
    w_max = max(max(s["w"].max().item() for s in pe), max(s["w"].max().item() for s in raw))
    ax_w.axvline(t_surface, color=GREY, ls="--", lw=1)
    ax_w.text(t_surface, w_max * 1.05, " surface", fontsize=6.5, color="#6b7280", va="top")
    (w_pe,) = ax_w.plot(t, pe[0]["w"].numpy(), color=BLUE, lw=1.3)
    (w_raw,) = ax_w.plot(t, raw[0]["w"].numpy(), color=RED, lw=1.1)
    ax_w.set_xlim(NEAR, FAR)
    ax_w.set_ylim(0, w_max * 1.1)
    ax_w.set_title("weights $w_s$ on the + ray", fontsize=8.5)
    ax_w.set_xlabel("distance $t$ along the ray", fontsize=7.5)
    ax_w.tick_params(labelsize=7)
    counter = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9)

    def update(k: int):
        im_pe.set_data(_img(pe[k]["rgb"], res))
        im_raw.set_data(_img(raw[k]["rgb"], res))
        dm_pe.set_data(_img(pe[k]["depth"], res))
        dm_raw.set_data(_img(raw[k]["depth"], res))
        l_pe.set_data(its[: k + 1], p_pe[: k + 1])
        l_raw.set_data(its[: k + 1], p_raw[: k + 1])
        k_pe.set_data(its[: k + 1], q_pe[: k + 1])
        k_raw.set_data(its[: k + 1], q_raw[: k + 1])
        w_pe.set_ydata(pe[k]["w"].numpy())
        w_raw.set_ydata(raw[k]["w"].numpy())
        counter.set_text(
            f"step {its[k]}   ·   held-out PSNR: with encoding {p_pe[k]:.1f} dB, without {p_raw[k]:.1f} dB"
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_gif(fig, update, list(range(len(pe))), path, fps=4)
