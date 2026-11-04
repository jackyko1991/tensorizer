"""Animate the search and show the study: the mixture weights softmax(α/T) per edge as the temperature falls, the
discrete DAG they pick, its parameters as images, the written equation, accuracy and time against the number of
training images for the CNN and the tensor DAG (every seed a dot), and a few test digits."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from PIL import Image

from problems._common.plotting import ACCENT, export_plot_data
from problems._common.progress import report
from problems._common.runtime import fast_mode

from . import equation as eq

METHODS = {  # colour, marker, line style, label
    "cnn": (ACCENT[1], "o", "-", "CNN (2 conv + linear)"),
    "tensor_dag": (ACCENT[0], "s", "-", "searched tensor DAG"),
    "pca_lda": (ACCENT[2], "^", "--", "PCA + LDA (closed form)"),
    "nearest_class_mean": (ACCENT[3], "D", ":", "nearest class mean"),
}
NODE_XY = {0: (0.07, 0.5), 1: (0.33, 0.86), 2: (0.33, 0.14), 3: (0.6, 0.5)}
EDGE_LABEL_XY = {0: (0.13, 0.76), 1: (0.13, 0.24), 2: (0.40, 0.5), 3: (0.30, 0.56), 4: (0.52, 0.75), 5: (0.52, 0.25)}


def save_gif_shared_palette(fig, update, frames: list[int], path: Path, fps: int = 4) -> Path:
    """Every frame quantised with ONE palette built from a strip of frames, so colours do not drift between frames."""
    fig.set_dpi(80)
    shots = []
    for i, f in enumerate(frames):
        update(f)
        fig.canvas.draw()
        shots.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        report(i + 1, len(frames), "frames")
    export_plot_data(fig, path)
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


def _draw_dag(ax, arch: dict, title: str) -> None:
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    kept = dict(arch["edges"])
    for e, (j, i) in enumerate(eq.CELL_EDGES):
        (x0, y0), (x1, y1) = NODE_XY[j], NODE_XY[i]
        on = e in kept
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(
            arrowstyle="-|>", lw=1.8 if on else 0.8, color="#1f2937" if on else "#d1d5db", ls="-" if on else "--",
            shrinkA=14, shrinkB=14, connectionstyle="arc3,rad=0.12"))  # fmt: skip
        lx, ly = EDGE_LABEL_XY[e]
        ax.text(lx, ly, eq.SHORT[kept[e]] if on else "0", fontsize=10, ha="center", va="center",
                color="#111827" if on else "#9ca3af",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))  # fmt: skip
    for k, (x, y) in NODE_XY.items():
        ax.add_patch(plt.Circle((x, y), 0.055, fc="#dbeafe" if k else "#e5e7eb", ec="#1e3a8a", lw=1.5))
        ax.text(x, y, "$x$" if k == 0 else f"$z_{k}$", ha="center", va="center", fontsize=12)
    steps = [(0.86, "stack $z$\n" + eq.SHORT[arch["pool"]]), (0.5, eq.SHORT[arch["reduce"]]),
             (0.14, eq.SHORT[arch["head"]] + r" $\to\hat y$")]  # fmt: skip
    for y, txt in steps:
        ax.text(0.88, y, txt, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#b45309"))  # fmt: skip
    ax.annotate("", (0.79, 0.84), (0.66, 0.56), arrowprops=dict(arrowstyle="-|>", color="#b45309", lw=1.4))
    for y0, y1 in ((0.76, 0.58), (0.42, 0.24)):
        ax.annotate("", (0.88, y1), (0.88, y0), arrowprops=dict(arrowstyle="-|>", color="#b45309", lw=1.4))


def _head_image(show: dict) -> tuple[np.ndarray, str, str]:
    """The read-out as an image: for 'vec' the weights (or prototypes) of each digit as 7×7 maps per channel
    (rows x, z1, z2, z3; columns digits 0–9), otherwise the (10, D) matrix."""
    a, th = show["arch"], show["theta"]
    r = a["reduce"]
    M = (th[f"W_{r}"] if a["head"] == "linear" else th[f"P_{r}"]).detach().cpu().numpy()  # (10, D)
    name = "$W_{kj}$" if a["head"] == "linear" else "prototypes $p_k$"
    if r != "vec":
        return M, f"read-out {name} on {eq.SHORT[r]}", "feature $j$"
    C = eq.CHANNELS
    tiles = M.reshape(10, C, 7, 7)
    img = np.full((C * 8 - 1, 10 * 8 - 1), np.nan)
    for k in range(10):
        for c in range(C):
            img[c * 8 : c * 8 + 7, k * 8 : k * 8 + 7] = tiles[k, c]
    return img, f"read-out {name} as 7×7 maps", "digit $k$ (columns)"


def animate(rows: list[dict], sizes, show: dict, data: dict, need: dict, targets, path: Path) -> Path:
    hist = show["hist"]
    W_hist = np.stack([w.numpy() for w in hist["w"]])  # (S, 9, 23)
    mask = eq.family_mask().numpy()
    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[3.5, 2.8, 1.35], width_ratios=[1.0, 1.15, 1.05], hspace=0.55,
                          wspace=0.42, left=0.085, right=0.955, top=0.95, bottom=0.03)  # fmt: skip

    # α tensor: operators × edges, blank where the typing rule forbids the operator
    ax_a = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("Blues").with_extremes(bad="#f3f4f6")
    shown = lambda w: np.where(mask, w, np.nan).T  # noqa: E731
    im_a = ax_a.imshow(shown(W_hist[0]), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax_a.set_yticks(range(len(eq.OPS)), [eq.SHORT[o] for o in eq.OPS], fontsize=9.5)
    ticks = [rf"${j}\to{i}$" for j, i in eq.CELL_EDGES] + ["pool", "reduce", "head"]
    ax_a.set_xticks(range(len(eq.EDGES)), ticks, fontsize=10, rotation=50, ha="right", rotation_mode="anchor")
    ax_a.set_xlabel(r"edge $e$ ($z_j \to z_i$, then the read-out)", fontsize=10)
    ax_a.set_ylabel("operator $o$", fontsize=10)
    ax_a.grid(False)
    cb = fig.colorbar(im_a, ax=ax_a, fraction=0.06, pad=0.02)
    cb.set_label("weight $w_{eo}$ (dimensionless)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    title_a = ax_a.set_title("", fontsize=11)  # softmax(α/T) at the current step

    ax_d = fig.add_subplot(gs[0, 1])
    from .solve import discretise  # the same rule as the search uses

    # parameters as images
    sub = gs[0, 2].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.6)
    ax_h = fig.add_subplot(sub[0])
    img, ttl, xl = _head_image(show)
    v = float(np.nanmax(np.abs(img))) or 1.0
    im_h = ax_h.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    ax_h.set_title(ttl, fontsize=11)
    ax_h.set_xlabel(xl, fontsize=10)
    if show["arch"]["reduce"] == "vec":
        ax_h.set_xticks(np.arange(10) * 8 + 3, [str(k) for k in range(10)], fontsize=10)
        ax_h.set_yticks(np.arange(eq.CHANNELS) * 8 + 3, ["$x$", "$z_1$", "$z_2$", "$z_3$"], fontsize=10)
        ax_h.set_ylabel("channel", fontsize=10)
    else:
        ax_h.set_yticks(range(10), [str(k) for k in range(10)], fontsize=9)
        ax_h.set_ylabel("digit $k$", fontsize=10)
    ax_h.grid(False)
    cbh = fig.colorbar(im_h, ax=ax_h, fraction=0.05, pad=0.02)
    cbh.set_label("weight (a.u.)", fontsize=10)
    cbh.ax.tick_params(labelsize=9)
    ax_k = fig.add_subplot(sub[1])
    convs = [e for e, op in show["arch"]["edges"] if op == "conv3"]
    if convs:
        K = show["theta"]["K"].detach().cpu().numpy()
        strip = np.full((3, 4 * len(convs) - 1), np.nan)
        for i, e in enumerate(convs):
            strip[:, 4 * i : 4 * i + 3] = K[e]
        vk = float(np.nanmax(np.abs(strip)))
        im_k = ax_k.imshow(strip, cmap="RdBu_r", vmin=-vk, vmax=vk, interpolation="nearest")
        ax_k.set_xticks([4 * i + 1 for i in range(len(convs))], [f"$K_{{{e + 1}}}$" for e in convs], fontsize=10)
        ax_k.set_yticks([])
        ax_k.set_title(f"learned 3×3 filters ({9 * len(convs)} numbers)", fontsize=10)
        cbk = fig.colorbar(im_k, ax=ax_k, fraction=0.05, pad=0.02)
        cbk.set_label("weight (a.u.)", fontsize=9)
        cbk.ax.tick_params(labelsize=8)
        ax_k.grid(False)
    else:
        ax_k.axis("off")
        ax_k.text(0.5, 0.5, "no learned filter chosen:\nevery cell edge is parameter-free", ha="center", va="center",
                  fontsize=10)  # fmt: skip

    # accuracy and time against n
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    for m, (col, mk, ls, lab) in METHODS.items():
        acc = np.array([[r["acc"] for r in rows if r["n"] == n and r["method"] == m] for n in sizes])
        jit = np.exp(rng.uniform(-0.06, 0.06, acc.shape))
        ax_acc.scatter((np.array(sizes)[:, None] * jit).ravel(), acc.ravel(), s=22, color=col, marker=mk, alpha=0.55,
                       lw=0)  # fmt: skip
        ax_acc.plot(sizes, acc.mean(1), ls, color=col, marker=mk, ms=6, lw=2, label=lab)
        if m in ("cnn", "tensor_dag"):
            t = np.array([[r["seconds"] for r in rows if r["n"] == n and r["method"] == m] for n in sizes])
            ax_t.scatter((np.array(sizes)[:, None] * jit).ravel(), t.ravel(), s=22, color=col, marker=mk, alpha=0.55,
                         lw=0)  # fmt: skip
            ax_t.plot(sizes, t.mean(1), ls, color=col, marker=mk, ms=6, lw=2, label=lab)
    ax_acc.axhline(0.1, color="#9ca3af", lw=1, ls="--")
    ax_acc.text(sizes[-1], 0.12, "chance", fontsize=9, ha="right", color="#6b7280")
    ax_acc.set_xscale("log")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_xlabel("training images per digit $n$ (images)", fontsize=10)
    ax_acc.set_ylabel("test accuracy (dimensionless)", fontsize=10)
    ax_acc.set_title("accuracy on 2000 test digits, every seed a dot", fontsize=11)
    ax_acc.legend(fontsize=9, loc="lower right")
    ax_acc.tick_params(labelsize=10)
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ts = [r["seconds"] for r in rows if r["seconds"] != ""]
    ax_t.set_ylim(min(ts) / 1.5, max(ts) * 1.5)
    ax_t.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax_t.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_t.yaxis.set_minor_formatter(NullFormatter())
    ax_t.set_xlabel("training images per digit $n$ (images)", fontsize=10)
    ax_t.set_ylabel("training wall time (s)", fontsize=10)
    ax_t.set_title("compute: search + fine-tune vs CNN training", fontsize=11)
    ax_t.legend(fontsize=9, loc="upper left")
    ax_t.tick_params(labelsize=10)
    lines = []
    for t in targets:
        f = lambda v: "never" if np.isnan(v) else f"{v:.0f}"  # noqa: E731
        lines.append(f"{t:.0%}: CNN {f(need['cnn', t])}, DAG {f(need['tensor_dag', t])}, "
                     f"LDA {f(need['pca_lda', t])}")  # fmt: skip
    ax_t.text(0.98, 0.04, "images per digit to reach\n" + "\n".join(lines), transform=ax_t.transAxes, fontsize=9,
              ha="right", va="bottom", bbox=dict(fc="white", ec="#d1d5db"))  # fmt: skip

    # a few test digits with both predictions
    sub = gs[1, 2].subgridspec(3, 4, hspace=0.7, wspace=0.1, height_ratios=[0.25, 1, 1])
    ax_lbl = fig.add_subplot(sub[0, :])
    ax_lbl.axis("off")
    ax_lbl.text(0.5, 0.0, f"test digits (models trained with n = {show['n']})", ha="center", va="bottom", fontsize=11)
    x = data["x_test"].cpu().numpy()[:, 0]
    y = data["y_test"].cpu().numpy()
    idx = np.arange(0, len(y), len(y) // 8)[:8] + 7
    for k, i in enumerate(idx):
        ax = fig.add_subplot(sub[1 + k // 4, k % 4])
        ax.imshow(x[i], cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        pd, pc = int(show["pred"][i]), int(show["cnn_pred"][i])
        ax.set_title(f"DAG {pd}{'✓' if pd == y[i] else '✗'}\nCNN {pc}{'✓' if pc == y[i] else '✗'}", fontsize=9.5)

    # the written equation
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.axis("off")
    eqs = show["equation"]
    ax_e.text(0.0, 1.0, f"the found equation (n = {show['n']} per digit, seed 0; test accuracy {show['acc']:.3f}, "
              f"{show['params']} learned numbers):", fontsize=11, va="top", transform=ax_e.transAxes)  # fmt: skip
    for k, line in enumerate(eqs):
        col, row = divmod(k, 3)
        ax_e.text(0.01 + 0.5 * col, 0.66 - 0.3 * row, f"${line}$", fontsize=12, va="top", transform=ax_e.transAxes)

    snaps = list(range(len(W_hist)))
    if fast_mode():
        snaps = snaps[:: max(1, len(snaps) // 3)][:3] + [snaps[-1]]
    hold, n_snap = 2, len(snaps)
    frames = list(range(n_snap * hold + 8))

    def update(f: int):
        k = snaps[min(f // hold, n_snap - 1)]
        im_a.set_data(shown(W_hist[k]))
        title_a.set_text(rf"softmax$(\alpha/T)$: step {hist['step'][k]}, $T$ = {hist['temp'][k]:.3f}")
        arch = discretise(hist["w"][k])
        _draw_dag(ax_d, arch, "argmax DAG now" if k < snaps[-1] else "the discrete DAG kept")

    return save_gif_shared_palette(fig, update, frames, path)
