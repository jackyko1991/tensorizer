"""Animate training: a training view and a held-out view (truth vs render), the splats as ellipses, loss and PSNR."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import EllipseCollection

from problems._common.plotting import save_gif

BLUE, RED, GREY = "#2563eb", "#dc2626", "#9ca3af"


def _img(x: torch.Tensor, res: int) -> np.ndarray:
    return x.reshape(res, res, 3).clamp(0, 1).numpy()


def _ellipses(snap: dict, res: int) -> EllipseCollection:
    """1σ ellipses of the 2D splats Σ' in the held-out view, filled with each Gaussian's colour and opacity."""
    lam, vec = torch.linalg.eigh(snap["cov2"])  # (M, 2), (M, 2, 2) ascending eigenvalues
    keep = snap["visible"] & (snap["mu2"] > -5).all(-1) & (snap["mu2"] < res + 5).all(-1)
    lam, vec, mu = lam[keep], vec[keep], snap["mu2"][keep]
    angle = torch.rad2deg(torch.atan2(vec[:, 1, 1], vec[:, 0, 1]))  # direction of the major axis
    rgba = torch.cat([snap["colour"][keep], (0.15 + 0.6 * snap["opacity"][keep])[:, None]], dim=1)
    return EllipseCollection(
        widths=(2 * lam[:, 1].sqrt()).numpy(), heights=(2 * lam[:, 0].sqrt()).numpy(), angles=angle.numpy(),
        units="xy", offsets=mu.numpy(), facecolors=rgba.numpy(), edgecolors=(0.2, 0.2, 0.2, 0.35), linewidths=0.4,
    )  # fmt: skip


def animate(gt_train: torch.Tensor, gt_novel: torch.Tensor, hist: dict, res: int, path: Path) -> Path:
    snaps, loss = hist["snaps"], hist["loss"].numpy()
    its = np.array([s["it"] for s in snaps])
    p_train = np.array([s["psnr_train"] for s in snaps])
    p_novel = np.array([s["psnr_novel"] for s in snaps])
    gt_n = _img(gt_novel, res)

    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.8))
    titles = ["training view: truth", "training view: splats", "held-out view: truth", "held-out view: splats",
              "held-out: 1σ ellipses of Σ'", "held-out: |error|"]  # fmt: skip
    ims = []
    for k, ax in enumerate(axes.flat[:6]):
        ax.set_title(titles[k], fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axes[0, 0].imshow(_img(gt_train, res), extent=(0, res, res, 0))
    ims.append(axes[0, 1].imshow(_img(snaps[0]["train"], res), extent=(0, res, res, 0)))
    axes[0, 2].imshow(gt_n, extent=(0, res, res, 0))
    ims.append(axes[0, 3].imshow(_img(snaps[0]["novel"], res), extent=(0, res, res, 0)))
    ax_el = axes[1, 0]
    ax_el.set_xlim(0, res)
    ax_el.set_ylim(res, 0)
    ax_el.set_aspect("equal")
    err = np.abs(_img(snaps[0]["novel"], res) - gt_n).mean(-1)
    ims.append(axes[1, 1].imshow(err, cmap="Greys", vmin=0, vmax=0.5, extent=(0, res, res, 0)))

    ax_l, ax_p = axes[1, 2], axes[1, 3]
    ax_l.semilogy(np.arange(1, len(loss) + 1), loss, color=GREY, lw=0.6, alpha=0.6)
    (loss_line,) = ax_l.plot([], [], color=BLUE, lw=1.3)
    ax_l.set_xlim(0, len(loss))
    ax_l.set_title("photometric loss", fontsize=8.5)
    ax_l.set_xlabel("Adam step", fontsize=7.5)
    ax_l.tick_params(labelsize=7)
    ax_l.minorticks_off()
    (pt_line,) = ax_p.plot([], [], "o-", color=BLUE, ms=2.5, lw=1.2, label="training views")
    (pn_line,) = ax_p.plot([], [], "s-", color=RED, ms=2.5, lw=1.2, label="held-out view")
    ax_p.set_xlim(0, its[-1])
    ax_p.set_ylim(min(p_train.min(), p_novel.min()) - 1, max(p_train.max(), p_novel.max()) + 2)
    ax_p.set_title("PSNR (dB)", fontsize=8.5)
    ax_p.set_xlabel("Adam step", fontsize=7.5)
    ax_p.tick_params(labelsize=7)
    ax_p.legend(fontsize=6.5, loc="lower right")
    counter = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9)
    state = {"coll": None}

    def update(k: int):
        s = snaps[k]
        ims[0].set_data(_img(s["train"], res))
        ims[1].set_data(_img(s["novel"], res))
        ims[2].set_data(np.abs(_img(s["novel"], res) - gt_n).mean(-1))
        if state["coll"] is not None:
            state["coll"].remove()
        state["coll"] = _ellipses(s, res)
        state["coll"].set_offset_transform(ax_el.transData)
        ax_el.add_collection(state["coll"])
        upto = max(1, s["it"])
        loss_line.set_data(np.arange(1, upto + 1), loss[:upto])
        pt_line.set_data(its[: k + 1], p_train[: k + 1])
        pn_line.set_data(its[: k + 1], p_novel[: k + 1])
        counter.set_text(
            f"step {s['it']}   ·   PSNR training {s['psnr_train']:.1f} dB   ·   held-out {s['psnr_novel']:.1f} dB"
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_gif(fig, update, list(range(len(snaps))), path, fps=4)
