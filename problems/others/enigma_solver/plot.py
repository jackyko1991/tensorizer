"""Animate the search (score heatmap per rotor order, best score per order) and then the decrypted text appearing."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from . import tensors

LINE = 76  # letters per text line
TICKS = [0, 5, 10, 15, 20, 25]


def animate(heat, best, orders, previews, cipher, plain, best_i, best_start, method, path: Path) -> Path:
    """heat (O, 26, 26) best score over the left start for every (mid, right) start; best (O,) per order."""
    heat, best = heat.cpu().numpy(), best.cpu().numpy()
    n_orders, n_reveal = len(orders), 30
    every = math.ceil(n_orders / 30)  # at most 30 search frames: with 60 orders, 2 orders per frame
    events = [("search", i) for i in [*range(every - 1, n_orders - 1, every), n_orders - 1]]
    events += [("text", k) for k in range(1, n_reveal + 1)] + [("text", n_reveal)] * 6  # then hold the final text
    lo, hi = np.percentile(heat, 50), heat.max()

    fig = plt.figure(figsize=(9, 5.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1, 1.5], hspace=0.45, wspace=0.3)
    ax_h, ax_b, ax_t = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])

    image = ax_h.imshow(heat[0], cmap=plt.get_cmap("viridis", 10), vmin=lo, vmax=hi, origin="lower")
    ax_h.set_xticks(TICKS, [tensors.ALPHABET[i] for i in TICKS])
    ax_h.set_yticks(TICKS, [tensors.ALPHABET[i] for i in TICKS])
    ax_h.set_xlabel("right rotor start")
    ax_h.set_ylabel("middle rotor start")
    ax_h.grid(False)
    (marker,) = ax_h.plot([], [], "o", ms=18, mfc="none", mec="#f59e0b", mew=2)
    title_h = ax_h.set_title("", fontsize=10)

    x = np.arange(n_orders)
    bars = ax_b.bar(x, np.zeros(n_orders), color="#9ca3af", width=0.8)
    if method == "ioc":
        ref = {"random 1/26": 1 / 26, "English": 0.066}
        ax_b.set_ylabel("index of coincidence")
    else:
        p = np.array(tensors.ENGLISH_FREQ) / sum(tensors.ENGLISH_FREQ)
        ref = {"random": np.log(p).mean(), "English": (p * np.log(p)).sum()}
        ax_b.set_ylabel("log-likelihood / letter")
    for (label, y), color in zip(ref.items(), ["#9ca3af", "#16a34a"]):
        ax_b.axhline(y, color=color, ls="--", lw=1)
        ax_b.text(n_orders - 0.5, y, label, ha="right", va="bottom", fontsize=8, color=color)
    span = max(best.max(), max(ref.values())) - min(best.min(), min(ref.values()))
    ax_b.set_ylim(min(best.min(), min(ref.values())) - 0.15 * span, max(best.max(), max(ref.values())) + 0.15 * span)
    ax_b.set_xlim(-0.8, n_orders - 0.2)
    ax_b.set_axisbelow(True)
    if n_orders <= 12:
        ax_b.set_xticks(x, ["-".join(o) for o in orders], fontsize=8)
    else:
        ax_b.set_xticks([])
        ax_b.set_xlabel(f"rotor order ({n_orders} orders)")
    title_b = ax_b.set_title("best score over all 26³ starts", fontsize=10)
    best_label = ax_b.text(best_i, 0, "", ha="center", va="bottom", fontsize=8, color="#1e40af")

    ax_t.axis("off")
    text = ax_t.text(0, 1, "", family="monospace", fontsize=9.5, va="top", transform=ax_t.transAxes)
    wrap = lambda s: "\n".join(s[i : i + LINE] for i in range(0, len(s), LINE))  # noqa: E731
    cipher_head = wrap(cipher[: 2 * LINE])

    def update(f: int):
        kind, k = events[f]
        done = k + 1 if kind == "search" else n_orders  # orders scored so far
        shown = k if kind == "search" else best_i
        leader = int(np.argmax(best[:done]))
        image.set_data(heat[shown])
        title_h.set_text(f"{'-'.join(orders[shown])}: best over left start")
        if shown == best_i and done > best_i:
            marker.set_data([best_start[2].item()], [best_start[1].item()])
        else:
            marker.set_data([], [])
        for i, bar in enumerate(bars):
            bar.set_height(best[i] if i < done else 0)
            bar.set_color("#2563eb" if i == leader else ("#f59e0b" if i == shown and kind == "search" else "#9ca3af"))
        if done > best_i:
            best_label.set_position((best_i, best[best_i]))
            best_label.set_text("-".join(orders[best_i]))
        title_b.set_text(f"best score over all 26³ starts — {done}/{n_orders} orders")
        if kind == "search":
            text.set_text(f"ciphertext\n{cipher_head}\n\nbest so far: {previews[leader]}")
        else:
            n = math.ceil(len(plain) * k / n_reveal)
            key = f"{'-'.join(orders[best_i])} at {tensors.to_text(torch.as_tensor(best_start))}"
            text.set_text(f"decrypted with rotors {key}\n{wrap(plain[:n])}")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.03)
    return save_gif(fig, update, frame_indices(len(events), 120), path, fps=8)
