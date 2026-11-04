"""Animate the four stages: the workflow DAG with the route of a test example, the router matrix G (fixed by the
workflow vs learned by a softmax gate), the training curves of every stage, and accuracy / size / time per model
with one dot per seed. Also a static figure (router confusion, accuracy vs bytes) and the CSV tables."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

from problems._common.plotting import ACCENT, save_png
from problems.ai.chain_of_thought.plot import save_gif_shared_palette

from . import tensors as T

MODELS = ["split", "global", "student", "int8"]
LABEL = {"split": "split", "global": "global", "student": "student", "int8": "int8", "moe": "MoE",
         "one_by_one": "1 by 1", "router": "gate + fresh experts",
         "router_given": "gate + workflow experts"}  # fmt: skip
COLOR = {"split": ACCENT[0], "global": ACCENT[3], "student": ACCENT[4], "int8": ACCENT[1], "moe": ACCENT[5],
         "one_by_one": "#6b7280", "router": ACCENT[6], "router_given": ACCENT[2]}  # fmt: skip
GREY = "#d1d5db"


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _segments(res: dict) -> list[tuple]:
    """Stages on one x axis: (name, offset, length) in the order they were trained."""
    cfg = res["cfg"]
    out, off = [], 0
    for name, n in (("split", cfg.split_steps), ("router", cfg.router_steps), ("global", cfg.global_steps),
                    ("student", cfg.student_steps)):  # fmt: skip
        out.append((name, off, max(n, 1)))
        off += max(n, 1)
    return out


def _box(ax, x, y, text, color, w=1.9, h=0.62, fc="white", lw=1.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", fc=fc, ec=color, lw=lw))
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color="#111827")


def _arrow(ax, a, b, color="#6b7280", lw=1.2):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2))


def draw_dag(ax, active: int | None) -> None:
    """The workflow: encode_image, encode_text → fuse_{op} (the input function picks one) → classify."""
    ax.clear()
    ax.set_xlim(-0.1, 10.1)
    ax.set_ylim(0.6, 5.4)
    ax.axis("off")
    _box(ax, 0.75, 4.6, "image $x$", "#6b7280", w=1.5)
    _box(ax, 0.75, 1.4, "text $t$", "#6b7280", w=1.5)
    _box(ax, 3.25, 4.6, "encode_image", ACCENT[0], w=2.5)
    _box(ax, 3.25, 1.4, "encode_text", ACCENT[5], w=2.5)
    _box(ax, 3.25, 3.0, "router $G$", "#111827", w=1.6, h=0.5, fc="#f3f4f6")
    for e, name in enumerate(T.OPS):
        y = 4.6 - 1.6 * e
        on = active == e
        _box(ax, 6.35, y, f"fuse_{name}", ACCENT[2] if on else GREY, w=2.3, fc="#ecfdf5" if on else "white",
             lw=2.4 if on else 1.2)  # fmt: skip
        for src in ((4.55, 4.6), (4.55, 1.4)):
            _arrow(ax, src, (5.15, y), color=ACCENT[2] if on else GREY, lw=1.6 if on else 0.8)
        _arrow(ax, (7.55, y), (8.4, 3.0), color=ACCENT[2] if on else GREY, lw=1.6 if on else 0.8)
    _arrow(ax, (1.55, 4.6), (1.95, 4.6))
    _arrow(ax, (1.55, 1.4), (1.95, 1.4))
    _arrow(ax, (3.25, 1.71), (3.25, 2.75), color="#111827")
    _box(ax, 9.2, 3.0, "classify", "#111827", w=1.5)
    ax.text(9.2, 2.25, r"$(d \circ k)\ \mathrm{mod}\ 10$", ha="center", fontsize=10)
    ax.set_title("workflow DAG = top-1 MoE with a fixed router", fontsize=11)


def animate(res: dict, path: Path, n_frames: int = 50, fps: int = 5) -> Path:
    segs = _segments(res)
    total = segs[-1][1] + segs[-1][2]
    ex = res["examples"]
    show = _np(res["show"])
    order = [int(i) for i in show[:: max(1, len(show) // 9)]][:9]  # examples of every operation
    G_fixed = _np(res["G_fixed"])
    M = res["M"]
    perm = res["router_perm"][M]  # fresh experts, seed 1: columns reordered to match the operations best
    gap = np.full((len(G_fixed), 1), np.nan)
    snaps = [np.concatenate([_np(s[0]), gap, _np(s[M])[:, list(perm)]], 1) for s in res["router_snaps"]]

    fig = plt.figure(figsize=(13.0, 7.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.0], hspace=0.6, wspace=0.55, left=0.06, right=0.99,
                          top=0.94, bottom=0.12)  # fmt: skip
    ax_dag = fig.add_subplot(gs[0, :2])
    ax_ex, ax_gf, ax_gl = (fig.add_subplot(gs[0, i]) for i in (2, 3, 4))
    ax_cur = fig.add_subplot(gs[1, :2])
    ax_acc, ax_size, ax_time = (fig.add_subplot(gs[1, i]) for i in (2, 3, 4))

    # router matrices
    cmap = ListedColormap(["white", ACCENT[2]]).with_extremes(bad="#9ca3af")
    ticks = np.arange(T.E)
    for ax, title in ((ax_gf, "$G$ fixed (workflow)"), (ax_gl, "$G$ learned (gate)")):
        if ax is ax_gf:
            ax.set_xticks(ticks, T.OP_SYMBOL, fontsize=10)
        else:
            ax.set_xticks([0, 1, 2, 4, 5, 6], list(T.OP_SYMBOL) + [f"$e_{q}$" for q in perm], fontsize=10)
        ax.set_xlabel("expert $e$ (index)", fontsize=10)
        ax.set_ylabel("test sample (index)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=10)
    ax_gf.imshow(G_fixed, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    im_gl = ax_gl.imshow(snaps[0], cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    rec = _np(res["router_recovery"])

    # training curves
    curves = res["curves"]
    lines = []  # (line, xs, ys)

    def add(xs, ys, color, ls="-", label=None):
        for m in range(ys.shape[1]):
            (ln,) = ax_cur.plot([], [], ls, color=color, lw=1.5, alpha=0.9, label=label if m == 0 else None)
            lines.append((ln, xs, ys[:, m]))

    off = {name: o for name, o, _ in segs}
    sp = curves["split"]
    xs = np.array([c[0] for c in sp]) + off["split"]
    add(xs, np.stack([_np(c[1]) for c in sp]), ACCENT[0], label="image (digit)")
    add(xs, np.stack([_np(c[2]) for c in sp]), ACCENT[5], label="text (number)")
    add(xs, np.stack([_np(c[3]) for c in sp]), ACCENT[2], label="fuse (table)")
    add(xs, np.stack([_np(c[4]) for c in sp]), "#111827", "--", label="chained")
    cv = curves["router"]
    xr, yr = np.array([c[0] for c in cv]) + off["router"], np.stack([_np(c[1]) for c in cv])  # (C, 2M)
    add(xr, yr[:, :M], COLOR["router_given"], label="gate, given experts")
    add(xr, yr[:, M:], COLOR["router"], label="gate, fresh experts")
    for name, color, label in (
        ("global", COLOR["global"], "global"),
        ("student", COLOR["student"], "student"),
    ):
        cv = curves[name]
        add(np.array([c[0] for c in cv]) + off[name], np.stack([_np(c[1]) for c in cv]), color, label=label)
    for name, o, n in segs:
        ax_cur.axvline(o, color=GREY, lw=1)
        ax_cur.text(o + n / 2, 1.03, {"split": "2 split", "router": "1b gate", "global": "3",
                                      "student": "4 distil"}[name], ha="center", fontsize=9)  # fmt: skip
    ax_cur.set_xlim(0, total)
    ax_cur.set_ylim(0, 1.62)
    ax_cur.set_yticks(np.linspace(0, 1, 6))
    ax_cur.set_xlabel("training step, stages in order (steps)", fontsize=10)
    ax_cur.set_ylabel("test accuracy (dimensionless)", fontsize=10)
    ax_cur.tick_params(labelsize=10)
    ax_cur.legend(fontsize=9, loc="upper center", ncol=3, framealpha=0.9, handlelength=1.5, columnspacing=0.8)

    # bars with one dot per seed
    acc = {k: _np(v) for k, v in res["acc"].items()}
    t_ms = {k: _np(v) * 1e3 for k, v in res["time"].items()}
    kb = {k: res["bytes"][k] / 1e3 for k in MODELS}
    lo = min(float(min(v.min() for v in acc.values() if v is not None)), 0.85)
    ready_at = {"split": off["split"] + segs[0][2], "global": off["global"] + segs[2][2],
                "student": total, "int8": total}  # fmt: skip
    time_models = ["one_by_one", "split", "moe", "student", "int8"]
    ready_at["moe"], ready_at["one_by_one"] = ready_at["split"], ready_at["split"]

    def bars(ax, keys, values, ylabel, title, ylim, now, log=False, dots=True):
        ax.clear()
        for i, k in enumerate(keys):
            if now < ready_at[k]:
                continue
            v = np.atleast_1d(values[k])
            ax.bar(i, v.mean(), color=COLOR[k], alpha=0.35, width=0.7)
            if dots:
                ax.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, "o", color=COLOR[k], ms=4)
        ax.set_xticks(range(len(keys)), [LABEL[k] for k in keys], rotation=40, ha="right", fontsize=10)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        if log:
            ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)

    tmax = max(float(v.max()) for v in t_ms.values())
    tmin = min(float(v.min()) for v in t_ms.values())

    def update(f):
        p = min(1.0, f / (0.8 * (n_frames - 1)))
        now = p * total
        for ln, xs_, ys_ in lines:
            keep = xs_ <= now + 1e-9
            ln.set_data(xs_[keep], ys_[keep])
        k = min(len(snaps) - 1, int(np.clip((now - off["router"]) / segs[1][2], 0, 1) * (len(snaps) - 1) + 1e-9))
        im_gl.set_data(snaps[k])
        ax_gl.set_title(f"learned: {rec[0]:.0%} | {rec[M]:.0%} match" if k == len(snaps) - 1 else "$G$ learned (gate)",
                        fontsize=11)  # fmt: skip
        i = order[(f // 5) % len(order)]
        op = int(ex["op"][i])
        draw_dag(ax_dag, op)
        ax_ex.clear()
        ax_ex.imshow(_np(ex["x"][i]).reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ax_ex.axis("off")
        ax_ex.set_title(f"“{T.words(ex['tok'][i])}”", fontsize=11)
        d, kk, y = int(ex["d"][i]), int(ex["k"][i]), int(ex["y"][i])
        ps, pq = int(ex["pred_global"][0, i]), int(ex["pred_int8"][0, i])
        ax_ex.text(0.5, -0.08, f"{d} {T.OP_SYMBOL[op]} {kk} mod 10 = {y}", transform=ax_ex.transAxes, ha="center",
                   va="top", fontsize=10)  # fmt: skip
        ax_ex.text(0.5, -0.22, f"DAG {ps} {'✓' if ps == y else '✗'}   int8 {pq} {'✓' if pq == y else '✗'}",
                   transform=ax_ex.transAxes, ha="center", va="top", fontsize=10)  # fmt: skip
        bars(ax_acc, MODELS, acc, "test accuracy (dimensionless)", "accuracy", (lo - 0.03, 1.0), now)
        bars(ax_size, MODELS, kb, "size (kB)", "weights", (0, max(kb.values()) * 1.15), now, dots=False)
        bars(ax_time, time_models, t_ms, "time per 1000 samples (ms)", "CPU, 1 thread", (tmin / 2, tmax * 2), now,
             log=True)  # fmt: skip
        return []

    update(0)
    return save_gif_shared_palette(fig, update, list(range(n_frames)), path, fps=fps)


def stages_png(res: dict, path: Path) -> Path:
    """Router confusion (true operation × chosen expert) for every seed, and accuracy against bytes."""
    conf = _np(res["router_confusion"])
    M = res["M"]
    panels = [0] + list(range(M, 2 * M))  # gate + workflow experts (seed 1), gate + fresh experts (every seed)
    n = len(panels)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.3 * (n + 1), 4.4), gridspec_kw={"width_ratios": [1] * n + [1.4]})
    for j, m in enumerate(panels):
        ax = axes[j]
        im = ax.imshow(conf[m], cmap="Greens", vmin=0, vmax=1 / T.E)
        for a in range(T.E):
            for b in range(T.E):
                c = "white" if conf[m, a, b] > 0.2 else "#111827"
                ax.text(b, a, f"{conf[m, a, b]:.2f}", ha="center", va="center", fontsize=10, color=c)
        ax.grid(False)
        ax.set_xticks(range(T.E), [f"$e_{q}$" for q in range(T.E)], fontsize=10)
        ax.set_yticks(range(T.E), T.OPS, fontsize=10)
        ax.set_xlabel("expert chosen by the gate (index)", fontsize=10)
        ax.set_ylabel("operation in the text (category)", fontsize=10)
        kind = "workflow experts" if m < M else "fresh experts"
        ax.set_title(f"gate + {kind}, seed {m % M + 1}\n{float(res['router_recovery'][m]):.0%} match", fontsize=11)
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("fraction of test samples (dimensionless)", fontsize=10)
        cb.ax.tick_params(labelsize=10)
    ax = axes[-1]
    for k in MODELS + ["router_given", "router"]:
        b = res["bytes"].get(k, res["bytes"]["split"]) / 1e3
        a = _np(res["acc"][k])
        ax.plot(np.full(len(a), b), a, "o", color=COLOR[k], ms=7, label=LABEL[k])
    allb = [v / 1e3 for v in res["bytes"].values()]
    ax.set_xlim(0, max(allb) * 1.1)
    alla = np.concatenate([_np(v) for v in res["acc"].values()])
    ax.set_ylim(max(0.0, alla.min() - 0.05), 1.0)
    ax.set_xlabel("stored weights (kB)", fontsize=10)
    ax.set_ylabel("test accuracy (dimensionless)", fontsize=10)
    ax.set_title("accuracy vs size (one dot per seed)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    return save_png(fig, path)


def write_tables(res: dict, out_dir: Path) -> None:
    M = res["M"]
    with open(out_dir / "stages.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["stage", "seed", "accuracy", "params", "bytes", "time_ms_per_1000"])
        for k in ["split", "router_given", "router", "global", "student", "int8"]:
            for m in range(M):
                t = res["time"].get(k)
                wr.writerow([k, m + 1, f"{float(res['acc'][k][m]):.4f}", res["params"].get(k, res["params"]["split"]),
                             res["bytes"].get(k, res["bytes"]["split"]),
                             f"{float(t[m]) * 1e3:.3f}" if t is not None else ""])  # fmt: skip
    eqv = res["equivalence"]
    with open(out_dir / "equivalence.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["quantity", "value"])
        for dt in ("float64", "float32"):
            for key, v in eqv[dt].items():
                wr.writerow([f"{dt}_{key}", v])
        for rule, (need, n) in eqv["jobs"].items():
            wr.writerow([f"jobs_needed_{rule}", f"{need} of {n}"])
        for k in ("split", "moe", "one_by_one"):
            for m in range(M):
                wr.writerow([f"time_ms_per_1000_{k}_seed{m + 1}", f"{float(res['time'][k][m]) * 1e3:.3f}"])
        for m in range(M):
            wr.writerow([f"router_given_experts_identity_match_seed{m + 1}", f"{float(res['router_identity'][m]):.4f}"])
            wr.writerow([f"router_fresh_experts_best_match_seed{m + 1}", f"{float(res['router_recovery'][M + m]):.4f}"])
