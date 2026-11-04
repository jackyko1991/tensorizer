"""Animate the experiment: the training DAG runs level by level while the per-stage loss curves grow; then the
accuracy of every stage against the monolith (seed dots), the sample-efficiency curves, the router G (fixed vs
learned) and test examples. A static PNG holds the measured schedule (serial vs critical path) and G.
Tables: stages.csv (accuracy per stage, seed and joint-set size) and timings.csv (seconds per rule)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from problems._common.plotting import ACCENT, export_plot_data, save_png
from problems._common.progress import report

from . import tensors as T

STAGE = {  # key: (label, colour)
    "connector_init": ("untrained conn.", "#9ca3af"),
    "connector": ("L1 connector", ACCENT[0]),
    "finetune": ("L2 fine-tune", ACCENT[2]),
    "router": ("Switch router", ACCENT[3]),
    "router_soft": ("dense router", ACCENT[6]),
    "monolith": ("monolith", ACCENT[1]),
}
RULE_COLOUR = {"train_cnn": ACCENT[5], "train_lm": ACCENT[4], "train_monolith": ACCENT[1],
               "train_connector": ACCENT[0], "global_finetune": ACCENT[2], "train_router": ACCENT[3],
               "train_router_soft": ACCENT[6], "evaluate": "#374151"}  # fmt: skip
SHORT = {"train_cnn": "CNN", "train_lm": "LM", "train_monolith": "monolith",
         "train_connector": "connector", "global_finetune": "fine-tune", "train_router": "Switch",
         "train_router_soft": "dense", "evaluate": "eval"}  # fmt: skip


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


def _node_xy(res: dict) -> dict:
    """Rule → (x, y): x = level, rules of one level stacked."""
    xy, lev = {}, res["levels"]
    for k in range(max(lev) + 1):
        group = [n for n, lv in zip(res["names"], lev) if lv == k]
        for i, n in enumerate(group):
            xy[n] = (k, (len(group) - 1) / 2 - i)
    return xy


def _draw_dag(ax, res: dict, done_level: float) -> None:
    ax.clear()
    xy = _node_xy(res)
    path = set(zip(res["path"][:-1], res["path"][1:]))
    A = res["A_rule"]
    for s, rs in enumerate(res["names"]):
        for r, rr in enumerate(res["names"]):
            if A[s][r]:
                crit = (rr, rs) in path
                (x0, y0), (x1, y1) = xy[rr], xy[rs]
                ax.annotate("", (x1 - 0.32, y1), (x0 + 0.32, y0),
                            arrowprops=dict(arrowstyle="-|>", lw=2.2 if crit else 1.0,
                                            color="#dc2626" if crit else "#9ca3af"))  # fmt: skip
    for n, lv in zip(res["names"], res["levels"]):
        x, y = xy[n]
        on = lv < done_level
        ax.text(x, y, f"{SHORT[n]}\n{res['seconds'][n]:.1f} s" if on else SHORT[n], ha="center", va="center",
                fontsize=9, color="white" if on else "#374151",
                bbox=dict(boxstyle="round,pad=0.3", fc=RULE_COLOUR[n] if on else "white", ec=RULE_COLOUR[n],
                          lw=1.5))  # fmt: skip
    ax.set_xlim(-0.55, max(res["levels"]) + 0.55)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks(range(max(res["levels"]) + 1))
    ax.set_yticks([])
    ax.set_xlabel("level $\\ell$ (dimensionless)", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_title(f"training DAG: critical path {res['makespan']:.0f} s, serial {res['serial']:.0f} s", fontsize=11)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def animate(res: dict, path: Path, n_frames: int = 36, hold: int = 10) -> Path:
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.45, 1.1, 1.3, 1.0], hspace=0.75, wspace=0.42,
                          left=0.04, right=0.985, top=0.93, bottom=0.10)  # fmt: skip
    ax_dag, ax_loss = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_bar, ax_eff = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])
    g_axes = [fig.add_subplot(a) for a in gs[1, 0:2].subgridspec(1, 3, wspace=0.12)]
    ex_axes = [fig.add_subplot(gs[1, 2 + i // 2].subgridspec(2, 1, hspace=0.9)[i % 2]) for i in range(4)]
    sizes = np.array(res["sizes"])
    S = len(res["seeds"])
    n_levels = max(res["levels"]) + 1
    loss = {k: v.numpy() for k, v in res["loss"].items()}
    lmax = max(float(np.nanmax(v)) for v in loss.values() if v.size)
    lmin = min(float(np.nanmin(v)) for v in loss.values() if v.size)
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.12, 0.12, S)

    # ---- static panels: G fixed vs learned
    ra = res["route_acc_show"]
    panels = ((res["G_fixed"], "fixed $G$\n(by data type)"),
              (res["G_learned"]["router"], f"Switch top-1\nroute acc. {ra['router']:.2f}"),
              (res["G_learned"]["router_soft"], f"dense-trained\nroute acc. {ra['router_soft']:.2f}"))  # fmt: skip
    for i, (ax, (G, title)) in enumerate(zip(g_axes, panels)):
        im = ax.imshow(G, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.axvline(T.K - 0.5, color="#dc2626", lw=1.5, ls="--")
        ax.set_xlabel("position $t$ (dimensionless)", fontsize=10)
        if i == 0:
            ax.set_ylabel("test example (dimensionless)", fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=10)
    cb = fig.colorbar(im, ax=g_axes, fraction=0.04, pad=0.02)
    cb.set_label("$p_{t,\\mathrm{img}}$, image expert (dimensionless)", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    def update(f: int) -> None:
        prog = min(1.0, f / max(1, n_frames - 1))  # fraction of the training run shown
        _draw_dag(ax_dag, res, prog * n_levels if f < n_frames else n_levels + 1)
        ax_loss.clear()
        for rule, v in loss.items():
            if not v.size:
                continue
            k = max(1, int(round(prog * len(v))))
            steps = np.arange(1, len(v) + 1)
            for m in range(v.shape[1]):
                ax_loss.plot(steps[:k], v[:k, m], color=RULE_COLOUR[rule], lw=0.6, alpha=0.35)
            ax_loss.plot(steps[:k], v[:k].mean(1), color=RULE_COLOUR[rule], lw=1.8, label=SHORT[rule])
        ax_loss.set_yscale("log")
        ax_loss.set_ylim(lmin * 0.8, lmax * 1.3)
        ax_loss.set_xlim(1, max(len(v) for v in loss.values()))
        ax_loss.set_xlabel("optimiser step (dimensionless)", fontsize=10)
        ax_loss.set_ylabel("loss $\\mathcal{L}$ (nats)", fontsize=10)
        ax_loss.set_title("per-stage loss, every model", fontsize=11)
        ax_loss.legend(fontsize=9, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.2), frameon=False,
                       handlelength=1.2, columnspacing=0.8)  # fmt: skip
        ax_loss.tick_params(labelsize=10)

        show = f >= n_frames
        ax_bar.clear()
        keys = list(STAGE)
        big = len(sizes) - 1
        for i, k in enumerate(keys):
            a = res["acc"][k][:, big].numpy()
            ax_bar.bar(i, a.mean() if show else 0, color=STAGE[k][1], width=0.7)
            if show:
                ax_bar.scatter(i + jit, a, s=18, color="black", zorder=3)
        if show:
            sym = res["symbolic"].numpy()
            ax_bar.axhline(sym.mean(), color="#6b7280", lw=1.2, ls=":", label="CNN label → text → LM")
            ax_bar.axhline(0.1, color="#9ca3af", lw=1.0, ls="--", label="chance")
            ax_bar.legend(fontsize=9, loc="upper left", framealpha=0.9)
        ax_bar.set_xticks(range(len(keys)), [STAGE[k][0] for k in keys], fontsize=9, rotation=35, ha="right",
                          rotation_mode="anchor")  # fmt: skip
        ax_bar.set_ylim(0, 1.18)
        ax_bar.set_xlim(-0.6, len(keys) - 0.4)
        ax_bar.set_ylabel("test accuracy (dimensionless)", fontsize=10)
        ax_bar.set_title(f"image + prompt → answer, $n$ = {sizes[big]}", fontsize=11)
        ax_bar.tick_params(axis="y", labelsize=10)

        ax_eff.clear()
        for k in ("connector", "finetune", "monolith"):
            a = res["acc"][k].numpy()  # (S, N)
            if show:
                for s in range(S):
                    ax_eff.scatter(sizes * (1 + 0.04 * jit[s]), a[s], s=14, color=STAGE[k][1], alpha=0.7)
                ax_eff.plot(
                    sizes, a.mean(0), "-o", color=STAGE[k][1], lw=1.8, ms=4, label=STAGE[k][0].replace("\n", " ")
                )
        ax_eff.axhline(0.1, color="#9ca3af", lw=1.0, ls="--")
        ax_eff.set_xscale("log")
        ax_eff.set_xlim(sizes.min() / 1.5, sizes.max() * 1.5)
        ax_eff.set_ylim(0, 1.05)
        ax_eff.set_xlabel("joint examples $n$ (dimensionless)", fontsize=10)
        ax_eff.set_ylabel("test accuracy (dimensionless)", fontsize=10)
        ax_eff.set_title("sample efficiency", fontsize=11)
        if show:
            ax_eff.legend(fontsize=9, loc="right", bbox_to_anchor=(1.0, 0.4), framealpha=0.9)
        ax_eff.tick_params(labelsize=10)

        for ax, ex in zip(ex_axes, res["examples"]):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(ex["img"], cmap="gray_r", extent=(0, 1, 0, 1))
            ax.set_xlim(0, 4.6)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(1.12, 0.78, f"[img]{ex['prompt']}", fontsize=10, va="center", family="monospace")
            if show:
                ok_d, ok_m = ex["dag"] == ex["truth"], ex["mono"] == ex["truth"]
                ax.text(1.12, 0.42, f"DAG: {ex['dag']}", fontsize=10, va="center", family="monospace",
                        color=ACCENT[2] if ok_d else ACCENT[1])  # fmt: skip
                ax.text(1.12, 0.08, f"mono: {ex['mono']}", fontsize=10, va="center", family="monospace",
                        color=ACCENT[2] if ok_m else ACCENT[1])  # fmt: skip
        fig.suptitle("Separately trained CNN + LM, stitched by one DAG edge vs a monolith trained end to end",
                     fontsize=13)  # fmt: skip

    return save_gif_shared_palette(fig, update, list(range(n_frames + hold)), path)


def schedule_png(res: dict, path: Path) -> Path:
    """Measured rule times as a Gantt chart: top, one worker running the rules level by level (serial); below, one
    worker per rule, each starting when its inputs are done (max-plus finish times); black edges = critical path."""
    names = res["names"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    t0 = 0.0
    top = len(names) + 0.6
    for i, n in enumerate(names):
        d, fin = res["seconds"][n], res["finish"][i]
        ax.barh(top, d, left=t0, height=0.7, color=RULE_COLOUR[n], edgecolor="white")
        crit = n in res["path"]
        ax.barh(len(names) - 1 - i, d, left=fin - d, height=0.6, color=RULE_COLOUR[n],
                edgecolor="#111827" if crit else "white", linewidth=2.0 if crit else 0.5)  # fmt: skip
        t0 += d
    labels = [SHORT[n] + (" (CP)" if n in res["path"] else "") for n in names][::-1] + ["serial"]
    ax.set_yticks(list(range(len(names))) + [top], labels, fontsize=10)
    ax.axhline(len(names) - 0.2, color="#9ca3af", lw=1, ls="--")
    ax.set_xlabel("wall time $t$ (s)", fontsize=10)
    ax.set_xlim(0, res["serial"] * 1.02)
    ax.set_ylim(-0.6, top + 0.6)
    ax.tick_params(labelsize=10)
    ax.set_title(f"measured schedule: serial {res['serial']:.0f} s; one worker per rule {res['makespan']:.0f} s "
                 "(critical path, CP)", fontsize=11)  # fmt: skip
    fig.tight_layout()
    return save_png(fig, path)


def write_tables(res: dict, out_dir: Path) -> None:
    with open(out_dir / "stages.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "seed", "n_joint", "test_accuracy"])
        for k in STAGE:
            for si, s in enumerate(res["seeds"]):
                for ni, n in enumerate(res["sizes"]):
                    w.writerow([k, s, n, f"{float(res['acc'][k][si, ni]):.4f}"])
        for si, s in enumerate(res["seeds"]):
            for k, ra in res["route_acc"].items():
                for ni, n in enumerate(res["sizes"]):
                    w.writerow([f"{k}_routing_accuracy", s, n, f"{float(ra[si, ni]):.4f}"])
            w.writerow(["cnn_alone_digit_accuracy", s, "", f"{float(res['cnn_acc'][si]):.4f}"])
            w.writerow(["lm_alone_text_prompts", s, "", f"{float(res['lm_acc'][si]):.4f}"])
            w.writerow(["symbolic_cnn_label_to_lm", s, "", f"{float(res['symbolic'][si]):.4f}"])
    with open(out_dir / "timings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rule", "level", "seconds", "finish_one_worker_per_rule_s", "on_critical_path", "models"])
        for i, n in enumerate(res["names"]):
            w.writerow([n, res["levels"][i], f"{res['seconds'][n]:.2f}", f"{res['finish'][i]:.2f}",
                        int(n in res["path"]), res["models"][n]])  # fmt: skip
