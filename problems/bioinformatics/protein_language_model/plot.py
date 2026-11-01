"""Animate training: the loss falls below the context-free baselines while the attention map sharpens into contacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import top_pairs


def animate(hist: dict, contacts: torch.Tensor, n_top: int, ce_uni: float, ce_prof: float, rand: float, path: Path):
    """hist from solve.train; contacts (L, L) bool; n_top predicted pairs are marked."""
    at = np.asarray(hist["at"])
    train = hist["train"].numpy()
    test, prec = np.asarray(hist["test"]), np.asarray(hist["precision"])
    scores, attn = hist["scores"].numpy(), hist["attention"].numpy()  # (S, L, L) each
    true = contacts.cpu().numpy()
    L = true.shape[0]
    win = max(1, len(train) // 40)
    smooth = np.convolve(train, np.ones(win) / win, mode="valid")  # running mean of the noisy batch loss
    ti, tj = np.nonzero(np.triu(true, 1))

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))
    (ax_loss, ax_prec), (ax_map, ax_att) = axes

    smooth_steps = np.arange(len(smooth)) + win - 1
    (train_line,) = ax_loss.plot([], [], color="#93c5fd", lw=1, label="train (batch)")
    (test_line,) = ax_loss.plot([], [], color=ACCENT[0], lw=2, label="test")
    ax_loss.axhline(ce_uni, color=ACCENT[1], ls="--", lw=1.2, label="unigram")
    ax_loss.axhline(ce_prof, color=ACCENT[4], ls="--", lw=1.2, label="per-site profile")
    ax_loss.set_xlim(0, at[-1])
    ax_loss.set_ylim(min(test.min(), ce_prof) - 0.1, ce_uni + 0.35)
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("masked-token loss (nats)")
    ax_loss.set_title("masked language modelling")
    ax_loss.legend(fontsize=7, loc="upper right")
    loss_dot = ax_loss.axvline(0, color="#f59e0b", lw=1)

    (prec_line,) = ax_prec.plot([], [], "o-", color=ACCENT[2], ms=3, lw=1.5, label="attention (APC)")
    ax_prec.axhline(rand, color="#9ca3af", ls="--", lw=1.2, label=f"random pairs ({rand:.2f})")
    ax_prec.set_xlim(0, at[-1])
    ax_prec.set_ylim(0, 1.05)
    ax_prec.set_xlabel("training step")
    ax_prec.set_ylabel(f"precision of top {n_top} pairs")
    ax_prec.set_title("contacts recovered from attention")
    ax_prec.legend(fontsize=7, loc="center right")

    cmap = plt.get_cmap("Blues", 12)
    for ax in (ax_map, ax_att):
        ax.set_xticks([0, L - 1], ["1", str(L)])
        ax.set_yticks([0, L - 1], ["1", str(L)])
        ax.grid(False)
    im_map = ax_map.imshow(np.zeros((L, L)), cmap=cmap, vmin=0, vmax=1)
    ax_map.scatter(tj, ti, s=28, facecolors="none", edgecolors="#374151", lw=1, label="planted contact")
    hits = ax_map.scatter([], [], s=12, color=ACCENT[2], label="top pair: hit")
    miss = ax_map.scatter([], [], s=12, color=ACCENT[1], marker="x", label="top pair: miss")
    ax_map.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    ax_map.set_title("APC attention score (lower) vs contacts (upper)", fontsize=9)
    im_att = ax_att.imshow(attn[0], cmap=plt.get_cmap("Purples", 12), vmin=0, vmax=np.percentile(attn[-1], 99))
    ax_att.set_xlabel("attended position $j$")
    ax_att.set_ylabel("query position $i$")
    title_att = ax_att.set_title("")

    def update(k: int):
        s = scores[k]
        off = s[~np.eye(L, dtype=bool)]
        lo, hi = np.percentile(off, 50), max(off.max(), 1e-9)
        shown = np.clip((s - lo) / (hi - lo), 0, 1)
        im_map.set_data(np.tril(shown, -1))  # score in the lower triangle, upper triangle left for the contacts
        pairs = top_pairs(torch.from_numpy(s), n_top).numpy()
        good = true[pairs[:, 0], pairs[:, 1]]
        hits.set_offsets(np.column_stack([pairs[good, 0], pairs[good, 1]]) if good.any() else np.empty((0, 2)))
        miss.set_offsets(np.column_stack([pairs[~good, 0], pairs[~good, 1]]) if (~good).any() else np.empty((0, 2)))
        im_att.set_data(attn[k])
        title_att.set_text(f"layer-1 attention $A_{{ij}}$, mean over heads (step {at[k]})")
        test_line.set_data(at[: k + 1], test[: k + 1])
        seen = smooth_steps <= at[k]
        train_line.set_data(smooth_steps[seen], smooth[seen])
        prec_line.set_data(at[: k + 1], prec[: k + 1])
        loss_dot.set_xdata([at[k], at[k]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(at), 40), path, fps=8)
