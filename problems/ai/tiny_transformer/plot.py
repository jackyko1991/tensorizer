"""Animate training: the loss falls from ln V, the samples turn from gibberish into Shakespeare-ish text, and the
causal attention maps of a fixed line take shape."""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, save_gif


def _label(ch: str) -> str:
    return {" ": "_", "\n": "\\n"}.get(ch, ch)


def animate(hist: dict, maps: np.ndarray, excerpt: str, heads: list[int], log_v: float, path: Path, hold: int = 3):
    """hist from solve.train; maps (checkpoints, layers, H, T, T) attention on the excerpt; heads[l] shown per layer."""
    at, val = np.asarray(hist["at"]), np.asarray(hist["val"])
    train = np.asarray(hist["train"])
    win = max(1, len(train) // 30)
    smooth = np.convolve(train, np.ones(win) / win, mode="valid")
    smooth_steps = np.arange(len(smooth)) + win - 1
    n_layers = maps.shape[1]
    labels = [_label(ch) for ch in excerpt]
    L = len(excerpt)

    fig = plt.figure(figsize=(9.8, 6.0))
    gs = fig.add_gridspec(2, 1 + n_layers, height_ratios=[1.6, 0.8], width_ratios=[1.25] + [1] * n_layers)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[1, :])
    ax_att = [fig.add_subplot(gs[0, 1 + layer]) for layer in range(n_layers)]

    (train_line,) = ax_loss.plot([], [], color="#93c5fd", lw=1.2, label="train (running mean)")
    (val_line,) = ax_loss.plot([], [], "o-", color=ACCENT[0], ms=3, lw=1.5, label="validation")
    ax_loss.axhline(log_v, color=ACCENT[1], ls="--", lw=1.2, label=f"ln V = {log_v:.2f} (uniform guess)")
    ax_loss.set_xlim(0, at[-1])
    ax_loss.set_ylim(min(val.min(), smooth.min()) - 0.2, log_v + 0.3)
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("next-character loss (nats)")
    ax_loss.legend(fontsize=7, loc="upper right")
    title = ax_loss.set_title("")

    ims = []
    for layer, ax in enumerate(ax_att):
        im = ax.imshow(maps[0, layer, heads[layer]], cmap=plt.get_cmap("Purples", 12), vmin=0, vmax=1)
        ims.append(im)
        ax.set_xticks(range(L), labels, fontsize=6.5, family="monospace")
        ax.set_yticks(range(L), labels, fontsize=6.5, family="monospace")
        ax.tick_params(length=0, pad=1)
        ax.grid(False)
        ax.set_title(f"layer {layer + 1}, head {heads[layer] + 1}: $A_{{ij}}$", fontsize=9)
        ax.set_xlabel("key $j$ (earlier characters)", fontsize=7)
    ax_att[0].set_ylabel("query $i$", fontsize=7)

    ax_text.axis("off")
    body = ax_text.text(0.0, 0.97, "", va="top", ha="left", family="monospace", fontsize=8.5,
                        transform=ax_text.transAxes)  # fmt: skip

    def update(frame: int):
        k = frame // hold
        seen = smooth_steps <= at[k]
        train_line.set_data(smooth_steps[seen], smooth[seen])
        val_line.set_data(at[: k + 1], val[: k + 1])
        title.set_text(f"step {at[k]}: validation loss {val[k]:.2f} nats")
        for layer, im in enumerate(ims):
            im.set_data(maps[k, layer, heads[layer]])
        lines = hist["text"][k].split("\n")[:9]
        wrapped = "\n".join(textwrap.shorten(s, 110, placeholder="…") if len(s) > 110 else s for s in lines)
        body.set_text(f"sample at step {at[k]} (temperature-scaled, same random seed every time):\n\n{wrapped}")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_gif(fig, update, list(range(len(at) * hold)), path, fps=6)
