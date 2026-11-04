"""Animate the experiment: training curves of direct vs CoT models, the accuracy-vs-n bars filling in, then a
CoT model writing its scratchpad one token at a time next to the attention of the query that writes it.
Also a static figure for experiments (b) and (c) and a CSV table of every accuracy."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from problems._common.plotting import ACCENT, export_plot_data
from problems._common.progress import report

from . import tensors

FORMAT_COLOR = {"direct": "#9ca3af", "CoT": ACCENT[0]}  # grey = no scratchpad, blue = chain of thought


def save_gif_shared_palette(fig, update, frames: list[int], path: Path, fps: int = 6) -> Path:
    """Render every frame, then quantise all of them with ONE palette so colours never drift between frames."""
    fig.set_dpi(80)
    shots = []
    for i, f in enumerate(frames):
        update(f)
        fig.canvas.draw()
        shots.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        report(i + 1, len(frames), "frames")
    export_plot_data(fig, path)  # numbers behind the final frame (Results page)
    plt.close(fig)
    sample = shots[:: max(1, len(shots) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * im.height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in shots]
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0)
    return tmp.replace(path)


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Running mean along axis 0 (same length, shorter window at the start)."""
    c = np.cumsum(np.concatenate([np.zeros((1, *x.shape[1:])), x]), axis=0)
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - win)
    return (c[i] - c[lo]) / (i - lo).reshape(-1, *[1] * (x.ndim - 1))


def animate(res: dict, path: Path, hold: int = 2) -> Path:
    cfg = res["cfg"]
    lengths = cfg.lengths
    L = len(lengths)
    at = res["at"].numpy()
    curve = res["curve_acc"].numpy().mean(-1)  # (C, L, 2) whole chain right, mean over seeds
    loss = _smooth(res["loss"].numpy().mean(-1), max(1, cfg.steps // 25))  # (steps, L, 2)
    acc = res["acc"].numpy()  # (L, 2, S) final greedy answer accuracy
    n_show = res["n_show"]
    tok = res["sample_tok"].tolist()
    T = 2 * n_show + 3
    A_all = res["sample_maps"].numpy()[..., :T, :T]  # (layers, H, T, T) on the sample string
    l_best, h_best = res["best_head"]
    A = A_all[l_best, h_best]

    fig = plt.figure(figsize=(11.0, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], width_ratios=[1, 1, 1])
    ax_loss, ax_acc, ax_bar = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_seq = fig.add_subplot(gs[1, :2])
    ax_att = fig.add_subplot(gs[1, 2])

    steps = np.arange(len(loss))
    loss_lines, acc_lines = {}, {}
    for li in range(L):
        for f in range(2):
            ls = "--" if f == 0 else "-"
            (loss_lines[li, f],) = ax_loss.plot([], [], ls, color=ACCENT[li % len(ACCENT)], lw=1.6)
            (acc_lines[li, f],) = ax_acc.plot([], [], ls, color=ACCENT[li % len(ACCENT)], lw=1.6, marker="o", ms=2.5)
    ax_loss.set_xlim(0, cfg.steps)
    ax_loss.set_ylim(0, max(0.8, float(loss.max()) * 1.05))
    ax_loss.set_xlabel("training step (steps)")
    ax_loss.set_ylabel("training loss (nats per trained token)")
    ax_loss.set_title("loss: dashed direct, solid CoT", fontsize=9)
    ax_acc.set_xlim(0, cfg.steps)
    ax_acc.set_ylim(-0.03, 1.03)
    ax_acc.axhline(0.5, color="#6b7280", lw=0.8, ls=":")
    ax_acc.set_xlabel("training step (steps)")
    ax_acc.set_ylabel("whole chain + answer right (fraction)")
    ax_acc.set_title("test strings, mean of seeds", fontsize=9)
    for li, n in enumerate(lengths):
        ax_acc.plot([], [], color=ACCENT[li % len(ACCENT)], lw=1.6, label=f"$n = {n}$")
    ax_acc.legend(fontsize=7, loc="center right")

    x = np.arange(L)
    width = 0.38
    bars = {}
    dots = {}
    for f, name in enumerate(["direct", "CoT"]):
        bars[f] = ax_bar.bar(x + (f - 0.5) * width, np.zeros(L), width * 0.92, color=FORMAT_COLOR[name], label=name)
        (dots[f],) = ax_bar.plot([], [], "o", color="#111827", ms=3.5, ls="none")
    ax_bar.axhline(0.5, color="#6b7280", lw=0.8, ls=":", label="chance")
    ax_bar.set_xticks(x, [f"{n}" + ("" if res["held_out"][i] else "*") for i, n in enumerate(lengths)])
    ax_bar.set_xlim(-0.6, L - 0.4)
    ax_bar.set_ylim(0, 1.25)
    ax_bar.set_xlabel("string length $n$ (bits)")
    ax_bar.set_ylabel("answer accuracy (fraction)")
    ax_bar.legend(fontsize=7, loc="upper center", ncol=3)
    bar_title = ax_bar.set_title("", fontsize=9)

    labels = [tensors.VOCAB[t] for t in tok[:T]]
    labels[-1] = str(res["sample_answer"])  # the model's own answer, not the true one
    im = ax_att.imshow(A, cmap=plt.get_cmap("Purples", 10), vmin=0, vmax=1)
    ax_att.set_xticks(range(T), labels, fontsize=6, family="monospace")
    ax_att.set_yticks(range(T), labels, fontsize=6, family="monospace")
    ax_att.tick_params(length=0, pad=1)
    ax_att.grid(False)
    ax_att.set_xlabel("key position $j$ (token)", fontsize=8)
    ax_att.set_ylabel("query position $i$ (token)", fontsize=8)
    cb = fig.colorbar(im, ax=ax_att, fraction=0.046, pad=0.03)
    cb.set_label("attention weight $A_{ij}$ (fraction)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    row_box = plt.Rectangle((-0.5, n_show - 0.5), T, 1, fill=False, ec=ACCENT[1], lw=1.4)
    key_box = plt.Rectangle((-0.5, n_show - 0.5), 1, 1, fill=False, ec=ACCENT[4], lw=1.6)
    ax_att.add_patch(row_box)
    ax_att.add_patch(key_box)
    att_title = ax_att.set_title("", fontsize=8)

    ax_seq.axis("off")
    ax_seq.set_xlim(0, 1)
    ax_seq.set_ylim(0, 1)
    cell = 0.92 / T
    boxes, texts = [], []
    for j in range(T):
        xj = 0.04 + j * cell
        r = plt.Rectangle((xj, 0.46), cell * 0.9, 0.2, fc="white", ec="#374151", lw=0.8)
        ax_seq.add_patch(r)
        boxes.append(r)
        texts.append(ax_seq.text(xj + cell * 0.45, 0.56, "", ha="center", va="center", family="monospace", fontsize=10))
    seq_title = ax_seq.text(0.0, 0.95, "", fontsize=9, va="top")
    seq_note = ax_seq.text(0.0, 0.3, "", fontsize=8.5, va="top", color="#374151")

    n_train = len(at)
    n_write = n_show + 2  # write s_1 … s_n, then the answer, then a last view
    frames = list(range((n_train + n_write) * hold))

    def show_sequence(k_written: int):
        """Bits and ":" always; s_1 … s_k written by the model; "=" and the answer once k = n + 1."""
        for j in range(T):
            visible = j <= n_show or (n_show < j <= n_show + k_written) or (k_written > n_show and j >= 2 * n_show + 1)
            texts[j].set_text(labels[j] if visible else "")
            scratch = n_show < j <= 2 * n_show
            fill = "#dbeafe" if scratch and visible else ("#f3f4f6" if j == 2 * n_show + 2 else "white")
            current = scratch and j == n_show + k_written and k_written <= n_show  # the cell being written
            boxes[j].set_facecolor(fill)
            boxes[j].set_edgecolor(ACCENT[1] if current else "#374151")

    def update(frame: int):
        c = frame // hold
        ci = min(c, n_train - 1)
        seen = steps <= at[ci]
        for (li, f), line in loss_lines.items():
            line.set_data(steps[seen], loss[seen, li, f])
        for (li, f), line in acc_lines.items():
            line.set_data(at[: ci + 1], curve[: ci + 1, li, f])
        final = c >= n_train - 1
        for f in range(2):
            heights = acc[:, f].mean(-1) if final else curve[ci, :, f]
            for li, b in enumerate(bars[f]):
                b.set_height(heights[li])
            if final:
                xs = np.repeat(x + (f - 0.5) * width, acc.shape[-1])
                dots[f].set_data(xs, acc[:, f].reshape(-1))
        bar_title.set_text("final answer accuracy, dots = seeds" if final else f"step {at[ci]}: whole chain right")
        if c < n_train:
            show_sequence(0)
            seq_title.set_text(f"CoT model (n = {n_show}) after training: the scratchpad it will write")
            seq_note.set_text("")
            k = 1
        else:
            k = min(c - n_train + 1, n_show + 1)
            show_sequence(k)
            seq_title.set_text("CoT model writes the running parity $s_k = s_{k-1} \\oplus b_k$ one token at a time")
            if k <= n_show:
                seq_note.set_text(f"step {k}: reads $b_{{{k}}}$ = {labels[k - 1]} and $s_{{{k - 1}}}$ → writes "
                                  f"$s_{{{k}}}$ = {labels[n_show + k]}")  # fmt: skip
            else:
                truth = res["sample_parity"]
                seq_note.set_text(f"answer p = {res['sample_answer']} (true parity {truth}): "
                                  "copied from the last scratchpad cell $s_n$")  # fmt: skip
        kk = min(k, n_show)
        row = n_show + kk - 1  # the query that writes s_k
        row_box.set_y(row - 0.5)
        key_box.set_x(kk - 1 - 0.5)
        key_box.set_y(row - 0.5)
        lh = np.unravel_index(A_all[:, :, row, kk - 1].argmax(), A_all.shape[:2])  # head that looks at b_k most
        im.set_data(A_all[lh])
        att_title.set_text(f"layer {lh[0] + 1}, head {lh[1] + 1}: query writing $s_{{{kk}}}$ (red row),\n"
                           f"key $b_{{{kk}}}$ (orange): $A$ = {A_all[lh][row, kk - 1]:.2f}")  # fmt: skip

    fig.tight_layout(h_pad=2.6)
    return save_gif_shared_palette(fig, update, frames, path, fps=4)


def experiments_png(res: dict, path: Path) -> Path:
    """(b) length generalisation and (c) accuracy against the number of written scratchpad steps."""
    cfg = res["cfg"]
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=90)
    if "gen_acc" in res:
        g = res["gen_acc"].numpy()  # (2, S, Lg)
        x = np.arange(len(cfg.gen_lengths))
        for f, name in enumerate(["direct", "CoT"]):
            xs = x + (f - 0.5) * 0.38
            ax_b.bar(xs, g[f].mean(0), 0.35, color=FORMAT_COLOR[name], label=name)
            ax_b.plot(np.repeat(xs, g.shape[1]).reshape(-1, g.shape[1]).T, g[f], "o", color="#111827", ms=3)
        ax_b.set_xticks(x, [str(n) for n in cfg.gen_lengths])
        inside = [i for i, n in enumerate(cfg.gen_lengths) if n <= cfg.train_max]
        if inside:
            label = f"trained on $n \\leq {cfg.train_max}$"
            ax_b.axvspan(-0.5, max(inside) + 0.5, color="#f3f4f6", zorder=0, label=label)
        ax_b.set_xlim(-0.5, len(x) - 0.5)
    ax_b.axhline(0.5, color="#6b7280", lw=0.8, ls=":")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_xlabel("test string length $n$ (bits)")
    ax_b.set_ylabel("answer accuracy (fraction)")
    ax_b.set_title("(b) length generalisation, dots = seeds", fontsize=10)
    ax_b.legend(fontsize=7, loc="upper right")
    n_big = max(cfg.lengths)
    for li, n in enumerate(cfg.lengths):
        h = res["hidden"][li, :, : n + 1].numpy().mean(0)
        ax_c.plot(np.arange(n + 1), h, "o-", color=ACCENT[li % len(ACCENT)], ms=3.5, lw=1.5, label=f"$n = {n}$")
    ax_c.axhline(0.5, color="#6b7280", lw=0.8, ls=":")
    ax_c.set_xlim(-0.3, n_big + 0.3)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_xlabel("scratchpad steps kept $k$ (tokens)")
    ax_c.set_ylabel("CoT answer accuracy (fraction)")
    ax_c.set_title("(c) later steps blanked out at test time", fontsize=10)
    ax_c.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_table(res: dict, path: Path) -> Path:
    """Every final accuracy, one row per (experiment, n, format, seed)."""
    cfg = res["cfg"]
    rows = [["experiment", "n", "format", "seed", "accuracy"]]
    for li, n in enumerate(cfg.lengths):
        for f, name in enumerate(["direct", "CoT"]):
            for s in range(cfg.seeds):
                rows.append(["a: fixed n", n, name, s, f"{res['acc'][li, f, s]:.4f}"])
    if "gen_acc" in res:
        for li, n in enumerate(cfg.gen_lengths):
            for f, name in enumerate(["direct", "CoT"]):
                for s in range(cfg.seeds):
                    rows.append([f"b: trained n<={cfg.train_max}", n, name, s, f"{res['gen_acc'][f, s, li]:.4f}"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    return path
