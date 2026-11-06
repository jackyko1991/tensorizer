"""The animation: the MNIST layout and the Swiss roll unrolling, epoch by epoch, from the spectral start; and the
static result panels: metrics against PCA and umap-learn, ρ/σ histograms, the fuzzy kNN graph, the final layouts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

from problems._common.plotting import export_plot_data, save_png
from problems._common.progress import report

from . import solve

FRAMES = 50  # layout snapshots per run (plus the spectral start)
RC = {"axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10}
DIGITS = plt.get_cmap("tab10")  # digit d → DIGITS(d), the same in every figure
ROLL = Normalize(1.5 * np.pi, 4.5 * np.pi)  # roll angle t → viridis
METHOD_COLOUR = {"PCA (SVD)": "#9ca3af", "tensor UMAP (forces)": "#2563eb", "tensor UMAP (autograd)": "#16a34a",
                 "umap-learn (offline)": "#dc2626"}  # fmt: skip


def save_gif_shared_palette(fig, update, frames: list, path: Path, durations: list[int]) -> Path:
    """Every frame quantised with ONE palette built from a strip of frames, so colours do not drift between frames;
    durations (ms) per frame let the last frame stay on screen."""
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
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=durations, loop=0)
    return tmp.replace(path)


def _limits(ys: list[np.ndarray]) -> tuple:
    """Axis limits covering every frame (5 % margin), fixed for the whole animation."""
    lo = np.min([y.min(0) for y in ys], axis=0)
    hi = np.max([y.max(0) for y in ys], axis=0)
    pad = 0.05 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def _digit_legend(ax, **kw):
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=DIGITS(d), label=str(d)) for d in range(10)]
    ax.legend(handles=handles, title="digit", **kw)


def animate(runs: dict, lab, t_roll, epochs: int, path: Path) -> Path:
    fm, fs = runs["mnist", "forces"]["frames"], runs["swiss_roll", "forces"]["frames"]
    ym = [f[1].numpy() for f in fm]
    ysr = [f[1].numpy() for f in fs]
    cm, cs = lab.cpu().numpy(), t_roll.cpu().numpy()
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13, 5.6))
        ax1 = fig.add_axes([0.06, 0.12, 0.36, 0.72])
        ax2 = fig.add_axes([0.55, 0.12, 0.36, 0.72])
        cax = fig.add_axes([0.925, 0.12, 0.015, 0.72])
        sc1 = ax1.scatter(ym[0][:, 0], ym[0][:, 1], c=[DIGITS(d) for d in cm], s=5, linewidths=0, label="digits")
        sc2 = ax2.scatter(ysr[0][:, 0], ysr[0][:, 1], c=cs, cmap="viridis", norm=ROLL, s=5, linewidths=0,
                          label="Swiss roll points")  # fmt: skip
        for ax, ys in ((ax1, ym), (ax2, ysr)):
            xl, yl = _limits(ys)
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_xlabel("$y_1$ (a.u.)")
            ax.set_ylabel("$y_2$ (a.u.)")
        _digit_legend(ax1, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10, title_fontsize=10,
                      handletextpad=0.2, borderaxespad=0.2)  # fmt: skip
        cb = fig.colorbar(sc2, cax=cax)
        cb.set_label("roll angle $t$ (rad)")
        sup = fig.suptitle("", fontsize=13)

        def update(k):
            ep, _, c1 = fm[k]
            sc1.set_offsets(ym[k])
            sc2.set_offsets(ysr[k])
            ax1.set_title(f"MNIST, {len(cm)} digits\ncross-entropy $C$ = {c1:.3g} (a.u.)")
            ax2.set_title(f"Swiss roll unrolling, {len(cs)} points\ncross-entropy $C$ = {fs[k][2]:.3g} (a.u.)")
            sup.set_text(f"UMAP as tensors: epoch {ep} / {epochs}" + ("  (spectral start)" if ep == 0 else ""))

        n = len(fm)
        durations = [1200] + [100] * (n - 2) + [3000] if n > 1 else [3000]
        return save_gif_shared_palette(fig, update, list(range(n)), path, durations)


def _bars(ax, rows, dataset, keys, names):
    meth = [r for r in rows if r["dataset"] == dataset]
    width = 0.8 / len(meth)
    for m, r in enumerate(meth):
        xs = np.arange(len(keys)) + (m - (len(meth) - 1) / 2) * width
        vals = [r.get(k, np.nan) for k in keys]
        ax.bar(xs, vals, width, color=METHOD_COLOUR[r["method"]], label=r["method"])
        for xx, v in zip(xs, vals):
            ax.text(xx, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9, rotation=90)
    ax.set_xticks(np.arange(len(keys)), names)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("score (dimensionless)")


def result_pngs(runs, rows, x, lab, xs, t_roll, ref, res: Path) -> None:
    with plt.rc_context(RC):
        # 1. metrics vs PCA vs umap-learn, and wall time
        fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
        _bars(axs[0], rows, "mnist", ["trust", "cont", "recall", "acc"],
              ["trustw.", "contin.", "recall@10", "10-NN acc."])  # fmt: skip
        axs[0].set_title("MNIST subset ($k$ = 10)")
        _bars(axs[1], rows, "swiss_roll", ["trust", "cont", "trust_flat", "recall_flat"],
              ["trustw.", "contin.", "trustw.\nvs flat", "recall@10\nvs flat"])  # fmt: skip
        axs[1].set_title("Swiss roll ($k$ = 10)")
        h, lb = axs[0].get_legend_handles_labels()
        fig.legend(h, lb, loc="upper center", ncol=4, fontsize=10, frameon=False)
        timed = [r for r in rows if r["seconds"] > 0]
        short = {"tensor UMAP (forces)": "tensor, forces", "tensor UMAP (autograd)": "tensor, autograd",
                 "umap-learn (offline)": "umap-learn"}  # fmt: skip
        labels = [f"{short[r['method']]}\n{'MNIST' if r['dataset'] == 'mnist' else 'Swiss roll'}" for r in timed]
        axs[2].barh(np.arange(len(timed)), [r["seconds"] for r in timed],
                    color=[METHOD_COLOUR[r["method"]] for r in timed])  # fmt: skip
        axs[2].set_yticks(np.arange(len(timed)), labels, fontsize=10)
        axs[2].invert_yaxis()
        axs[2].set_xlim(0, 1.2 * max(r["seconds"] for r in timed))
        axs[2].set_xlabel("wall time (s)")
        axs[2].set_title("time to embed (CPU)")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save_png(fig, res / "metrics.png")

        # 2. ρ and σ histograms
        fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
        for row, name in enumerate(("mnist", "swiss_roll")):
            o = runs[name, "forces"]
            for col, (key, sym) in enumerate((("rho", r"$\rho_i$"), ("sigma", r"$\sigma_i$"))):
                v = o[key].cpu().numpy()
                axs[row, col].hist(v, bins=40, color="#2563eb" if col == 0 else "#ea580c")
                axs[row, col].set_xlabel(f"{sym}, " + ("pixel-space distance (a.u.)" if name == "mnist"
                                                       else "distance (a.u.)"))  # fmt: skip
                axs[row, col].set_ylabel("points (count)")
                axs[row, col].set_title(f"{'MNIST' if name == 'mnist' else 'Swiss roll'}: {sym}, median "
                                        f"{np.median(v):.3g}")  # fmt: skip
        fig.tight_layout()
        save_png(fig, res / "rho_sigma.png")

        # 3. the fuzzy kNN graph on small samples
        fig = plt.figure(figsize=(13, 5.6))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        g = torch.Generator().manual_seed(0)
        pick = torch.randperm(len(xs), generator=g)[:300].to(xs.device)
        sub = xs[pick]
        gr = solve.fuzzy_graph(sub, 10)
        i, j = torch.nonzero(torch.triu(gr["W"]), as_tuple=True)
        p = sub.cpu().numpy()
        w = gr["W"][i, j].cpu().numpy()
        wn = Normalize(0, 1)
        lc3 = Line3DCollection(np.stack([p[i.cpu().numpy()], p[j.cpu().numpy()]], axis=1), cmap="magma_r", norm=wn,
                               linewidths=0.8)  # fmt: skip
        lc3.set_array(w)
        ax.add_collection(lc3)
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=t_roll[pick].cpu().numpy(), cmap="viridis", norm=ROLL, s=10,
                        label="sampled points")  # fmt: skip
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.12).set_label("roll angle $t$ (rad)")
        ax.set_xlabel("$x_1$ (a.u.)")
        ax.set_ylabel("$x_2$ (a.u.)")
        ax.set_zlabel("$x_3$ (a.u.)")
        ax.set_title("Swiss roll, 300 points: fuzzy 10-NN graph in 3D")
        ax2 = fig.add_subplot(1, 2, 2)
        o = runs["mnist", "forces"]
        y = o["Y"].cpu().numpy()
        ax2.scatter(y[:, 0], y[:, 1], c=[DIGITS(d) for d in lab.cpu().numpy()], s=3, alpha=0.35, linewidths=0,
                    label="all digits")  # fmt: skip
        src = torch.randperm(len(y), generator=g)[:40]
        W = o["W"].cpu()
        segs, ws = [], []
        for s_ in src.tolist():
            nb = torch.nonzero(W[s_]).flatten().tolist()
            segs += [(y[s_], y[t_]) for t_ in nb]
            ws += [float(W[s_, t_]) for t_ in nb]
        lc = LineCollection(segs, cmap="magma_r", norm=wn, linewidths=0.9)
        lc.set_array(np.array(ws))
        ax2.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax2)
        cb.set_label("membership $w_{ij}$ (dimensionless)")
        ax2.set_xlabel("$y_1$ (a.u.)")
        ax2.set_ylabel("$y_2$ (a.u.)")
        ax2.set_title("MNIST layout: all graph edges of 40 points")
        fig.tight_layout()
        save_png(fig, res / "knn_graph.png")

        # 4. final layouts side by side
        y_ours = runs["mnist", "forces"]["Y"].cpu()
        panels = [("tensor UMAP (forces)", y_ours), ("tensor UMAP (autograd)", runs["mnist", "autograd"]["Y"].cpu())]
        if ref is not None:
            yr, disp = solve.procrustes(torch.from_numpy(ref["mnist"]), y_ours)
            panels.append((f"umap-learn, Procrustes-aligned\n(disparity {disp:.2f})", yr))
        panels.append(("PCA (first two axes)", solve.pca(x).cpu()))
        fig, axs = plt.subplots(1, len(panels), figsize=(4.4 * len(panels) + 1, 4.9))
        c = [DIGITS(d) for d in lab.cpu().numpy()]
        for ax, (title, yy) in zip(axs, panels):
            ax.scatter(yy[:, 0].numpy(), yy[:, 1].numpy(), c=c, s=3, linewidths=0, label=title.split("\n")[0])
            ax.set_title(title)
            ax.set_xlabel("$y_1$ (a.u.)")
            ax.set_ylabel("$y_2$ (a.u.)")
        _digit_legend(axs[-1], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10, title_fontsize=10)
        fig.tight_layout()
        save_png(fig, res / "embeddings.png")
