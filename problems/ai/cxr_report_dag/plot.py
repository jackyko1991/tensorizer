"""Animate the experiment: the training DAG runs level by level while the per-stage loss curves grow; then the label
F1 of every stage against the monolith (seed dots), the caption metrics, the router G (fixed vs learned), and test
X-rays with the CNN's finding bars and the report typed out letter by letter, and PMC figures with their captions.
A static PNG holds the measured schedule (serial vs critical path).
Tables: stages.csv, captions.csv, caption_examples.csv, report_examples.csv and timings.csv."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from problems._common.plotting import ACCENT, save_png
from problems.ai.stitch_llm_cnn_dag.plot import save_gif_shared_palette

from . import tensors as T

STAGE = {  # key: (label, colour)
    "cnn": ("CNN alone", ACCENT[5]),
    "symbolic": ("CNN label→LM", "#6b7280"),
    "connector": ("L1 connector", ACCENT[0]),
    "finetune": ("L2 fine-tune", ACCENT[2]),
    "router": ("learned router", ACCENT[6]),
    "monolith": ("monolith", ACCENT[1]),
}
CAP = {"finetune": ("stage 1 only", ACCENT[2]), "caption": ("L3 captions", ACCENT[3]),
       "caption_shuffled": ("shuffled img.", "#9ca3af"), "constant": ("constant", "#374151")}  # fmt: skip
RULE_COLOUR = {"train_cnn": ACCENT[5], "train_lm": ACCENT[4], "train_monolith": ACCENT[1],
               "train_connector": ACCENT[0], "global_finetune": ACCENT[2], "train_router": ACCENT[6],
               "caption_finetune": ACCENT[3], "evaluate": "#374151"}  # fmt: skip
SHORT = {"train_cnn": "CNN", "train_lm": "LM", "train_monolith": "monolith", "train_connector": "conn.",
         "global_finetune": "fine-tune", "train_router": "router", "caption_finetune": "captions",
         "evaluate": "eval"}  # fmt: skip


def _node_xy(res: dict) -> dict:
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
    for s, rs in enumerate(res["names"]):
        for r, rr in enumerate(res["names"]):
            if res["A_rule"][s][r]:
                crit = (rr, rs) in path
                (x0, y0), (x1, y1) = xy[rr], xy[rs]
                ax.annotate("", (x1 - 0.36, y1), (x0 + 0.36, y0),
                            arrowprops=dict(arrowstyle="-|>", lw=2.2 if crit else 1.0,
                                            color="#dc2626" if crit else "#9ca3af"))  # fmt: skip
    for n, lv in zip(res["names"], res["levels"]):
        x, y = xy[n]
        on = lv < done_level
        ax.text(x, y, f"{SHORT[n]}\n{res['seconds'][n]:.0f} s" if on else SHORT[n], ha="center", va="center",
                fontsize=9, color="white" if on else "#374151",
                bbox=dict(boxstyle="round,pad=0.25", fc=RULE_COLOUR[n] if on else "white", ec=RULE_COLOUR[n],
                          lw=1.5))  # fmt: skip
    ax.set_xlim(-0.55, max(res["levels"]) + 0.55)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xticks(range(max(res["levels"]) + 1))
    ax.set_yticks([])
    ax.set_xlabel("level $\\ell$ (dimensionless)", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_title(f"training DAG: critical path {res['makespan']:.0f} s, serial {res['serial']:.0f} s", fontsize=11)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def _typed(text: str, frac: float) -> str:
    return text[: int(round(frac * len(text)))]


def _wrap(text: str, width: int = 26) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return "\n".join(lines + [cur] if cur else lines)


def animate(res: dict, path: Path, n_frames: int = 30, hold: int = 24) -> Path:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.05, 1.25, 1.05], wspace=0.4, left=0.04, right=0.985, top=0.93,
                          bottom=0.6)  # fmt: skip
    ax_dag, ax_loss = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax_f1, ax_cap = fig.add_subplot(gs[2]), fig.add_subplot(gs[3])
    bot = fig.add_gridspec(1, 3, width_ratios=[3.05, 0.9, 1.25], wspace=0.2, left=0.02, right=0.985, top=0.42,
                           bottom=0.06)  # fmt: skip
    ex_gs = bot[0].subgridspec(2, 2, hspace=0.6, wspace=0.06)
    ex_axes = []
    for i in range(4):  # image | gap | finding bars | (room for the finding names, right) | written reports
        sub = ex_gs[i // 2, i % 2].subgridspec(1, 5, width_ratios=[1.0, 0.1, 0.6, 1.08, 1.3], wspace=0.04)
        ex_axes.append((fig.add_subplot(sub[0]), fig.add_subplot(sub[2]), fig.add_subplot(sub[4])))
    g_axes = [fig.add_subplot(a) for a in bot[1].subgridspec(1, 2, wspace=0.3)]
    cap_gs = bot[2].subgridspec(2, 1, hspace=0.15)
    cap_axes = [fig.add_subplot(cap_gs[i]) for i in range(2)]
    S = len(res["seeds"])
    n_levels = max(res["levels"]) + 1
    loss = {k: v.numpy() for k, v in res["loss"].items() if v.numel()}
    lmax = max(float(np.nanmax(v)) for v in loss.values())
    lmin = min(float(np.nanmin(v)) for v in loss.values())
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, S)

    # ---- static: router G, fixed vs learned (top-1 choice of the learned router's p)
    panels = (
        (res["G_fixed"], "fixed $G$\n(by data type)"),
        (res["G_learned"], f"learned router\nroute acc. {res['route_show']:.2f}"),
    )
    for i, (ax, (G, title)) in enumerate(zip(g_axes, panels)):
        im = ax.imshow(G, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.axvline(T.K - 0.5, color="#dc2626", lw=1.5, ls="--")
        if i == 0:
            ax.set_ylabel("test image (dimensionless)", fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_xticks([0, 30])
    cb = fig.colorbar(im, ax=g_axes, fraction=0.05, pad=0.03)
    cb.set_label("$p_{t,\\mathrm{img}}$ (dimensionless)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    p0, p1 = g_axes[0].get_position(), g_axes[1].get_position()  # one x-label shared by both heatmaps
    fig.text(
        (p0.x0 + p1.x1) / 2, p0.y0 - 0.035, "input position $t$ (dimensionless)", ha="center", va="top", fontsize=10
    )

    def update(f: int) -> None:
        prog = min(1.0, f / max(1, n_frames - 1))
        show = f >= n_frames
        typed = min(1.0, (f - n_frames + 1) / max(1, hold * 0.7)) if show else 0.0
        _draw_dag(ax_dag, res, prog * n_levels if not show else n_levels + 1)

        ax_loss.clear()
        for rule, v in loss.items():
            k = max(1, int(round(prog * len(v))))
            steps = np.arange(1, len(v) + 1)
            ax_loss.plot(steps[:k], v[:k].mean(1), color=RULE_COLOUR[rule], lw=1.6, label=SHORT[rule])
        ax_loss.set_yscale("log")
        ax_loss.set_ylim(lmin * 0.7, lmax * 1.4)
        ax_loss.set_xlim(1, max(len(v) for v in loss.values()))
        ax_loss.set_xlabel("optimiser step (dimensionless)", fontsize=10)
        ax_loss.set_ylabel("loss $\\mathcal{L}$ (nats)", fontsize=10)
        ax_loss.set_title("per-stage loss (seed mean)", fontsize=11)
        ax_loss.legend(fontsize=9, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.2), frameon=False,
                       handlelength=1.0, columnspacing=0.7)  # fmt: skip
        ax_loss.tick_params(labelsize=10)

        ax_f1.clear()
        for i, k in enumerate(STAGE):
            a = res["f1_micro"][k].numpy()
            ax_f1.bar(i, a.mean() if show else 0, color=STAGE[k][1], width=0.7)
            if show:
                ax_f1.scatter(i + jit, a, s=16, color="black", zorder=3)
        ax_f1.axhline(res["chance"], color="#9ca3af", lw=1.0, ls="--", label="random at prevalence")
        ax_f1.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax_f1.set_xticks(range(len(STAGE)), [v[0] for v in STAGE.values()], fontsize=9, rotation=35, ha="right",
                         rotation_mode="anchor")  # fmt: skip
        ax_f1.set_ylim(0, 0.6)
        ax_f1.set_xlim(-0.6, len(STAGE) - 0.4)
        ax_f1.set_ylabel("micro $F_1$ of six findings (dimensionless)", fontsize=10)
        ax_f1.set_title("NIH test: labels parsed from the report", fontsize=11)
        ax_f1.tick_params(axis="y", labelsize=10)

        ax_cap.clear()
        keys = list(CAP)
        for j, (metric, hatch) in enumerate((("rouge_l", ""), ("kw_recall", "//"))):
            for i, k in enumerate(keys):
                a = res["cap"][k][metric].numpy()
                ax_cap.bar(i + (j - 0.5) * 0.38, a.mean() if show else 0, width=0.36, color=CAP[k][1],
                           hatch=hatch, edgecolor="white")  # fmt: skip
                if show:
                    ax_cap.scatter(i + (j - 0.5) * 0.38 + jit * 0.5, a, s=10, color="black", zorder=3)
        ax_cap.legend(handles=[Patch(fc="#9ca3af", label="ROUGE-L"),
                               Patch(fc="#9ca3af", hatch="//", ec="white", label="keyword recall")],
                      fontsize=9, loc="upper left", framealpha=0.9)  # fmt: skip
        ax_cap.set_xticks(range(len(keys)), [CAP[k][0] for k in keys], fontsize=9, rotation=30, ha="right",
                          rotation_mode="anchor")  # fmt: skip
        ax_cap.set_ylim(0, 0.45)
        ax_cap.set_ylabel("score (dimensionless)", fontsize=10)
        ax_cap.set_title("PMC test captions (stage 2)", fontsize=11)
        ax_cap.tick_params(axis="y", labelsize=10)

        for (a_img, a_bar, a_txt), ex in zip(ex_axes, res["examples"]):
            a_img.clear()
            a_img.imshow(ex["img"], cmap="gray", vmin=0, vmax=255)
            a_img.set_xticks([])
            a_img.set_yticks([])
            a_bar.clear()
            a_bar.barh(np.arange(6), ex["q"] if show else np.zeros(6), color=ACCENT[5])
            a_bar.axvline(0.5, color="#9ca3af", lw=0.8, ls="--")
            a_bar.set_xlim(0, 1)
            a_bar.set_ylim(5.6, -0.6)
            a_bar.yaxis.tick_right()
            a_bar.set_yticks(range(6), T.FINDINGS, fontsize=9)
            a_bar.set_xticks([0, 0.5, 1], ["0", ".5", "1"], fontsize=9)
            a_bar.set_xlabel("$q_f$ (dimensionless)", fontsize=9)
            a_txt.clear()
            a_txt.axis("off")
            a_txt.set_xlim(0, 1)
            a_txt.set_ylim(0, 1)
            if show:
                ok = ex["pred"] == ex["truth"]
                a_txt.text(0.1, 1.0, "DAG report:", fontsize=9, va="top", color="#374151")
                a_txt.text(0.1, 0.84, _wrap(_typed(ex["dag"], typed), 16), fontsize=9, va="top",
                           family="monospace", color=ACCENT[2] if ok else ACCENT[1])  # fmt: skip
                a_txt.text(0.1, 0.36, "monolith:", fontsize=9, va="top", color="#374151")
                a_txt.text(0.1, 0.20, _wrap(_typed(ex["mono"], typed), 16), fontsize=9, va="top",
                           family="monospace", color="#6b7280")  # fmt: skip
            a_img.set_title("true:\n" + _wrap(ex["truth_short"], 16), fontsize=9, loc="left")

        for ax, ex in zip(cap_axes, res["cap_examples"]):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(ex["img"], cmap="gray", vmin=0, vmax=255, extent=(0, 1, 0.55, 1.55))
            ax.set_xlim(0, 3.7)
            ax.set_ylim(-0.45, 1.58)
            ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(1.1, 1.52, "PMC text (start):\n" + _wrap(_clip(ex["truth"], 96), 32), fontsize=9, va="top",
                    color="#374151")  # fmt: skip
            if show:
                ax.text(1.1, 0.5, "model (start):\n" + _wrap(_clip(_typed(ex["gen"], typed), 96), 32),
                        fontsize=9, va="top",
                        family="monospace", color=ACCENT[3])  # fmt: skip
        fig.suptitle("Chest X-ray CNN + tiny report LM, trained apart and stitched by one DAG edge (a teaching toy, "
                     "not clinical report generation)", fontsize=13)  # fmt: skip

    # the first frame is the final state (an informative still), then the training replay, then the final hold
    return save_gif_shared_palette(fig, update, [n_frames + hold - 1] + list(range(n_frames + hold)), path)


def schedule_png(res: dict, path: Path) -> Path:
    """Measured rule times as a Gantt chart: one worker running the rules level by level (serial), and one worker
    per rule starting when its inputs are done (max-plus finish times); black edges = critical path."""
    names = res["names"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    t0, top = 0.0, len(names) + 0.6
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
        w.writerow(["stage", "seed", "micro_f1", "macro_f1", "wellformed_template", "exact_template"])
        for k in STAGE:
            for si, s in enumerate(res["seeds"]):
                r = [res[m][k][si] if k in res[m] else float("nan") for m in ("f1_micro", "f1_macro", "wellformed",
                                                                                "exact")]  # fmt: skip
                w.writerow([k, s] + [f"{float(v):.4f}" for v in r])
        for si, s in enumerate(res["seeds"]):
            w.writerow(["cnn_mean_auroc", s, f"{float(res['auc'][si].mean()):.4f}", "", "", ""])
            w.writerow(["router_routing_accuracy", s, f"{float(res['route_acc'][si]):.4f}", "", "", ""])
        w.writerow(["chance_micro_f1_random_at_prevalence", "", f"{res['chance']:.4f}", "", "", ""])
    with open(out_dir / "captions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "bleu1", "rouge_l", "keyword_recall", "keyword_precision",
                    "heldout_nats_per_letter"])  # fmt: skip
        for k, c in res["cap"].items():
            for si, s in enumerate(res["seeds"]):
                w.writerow([k, s] + [f"{float(c[m][si]):.4f}" for m in ("bleu1", "rouge_l", "kw_recall",
                                                                        "kw_precision", "nats")])  # fmt: skip
    with open(out_dir / "caption_examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmc_test_index", "true_caption_cleaned", "stage1_model", "caption_model", "shuffled_images"])
        for i, row in enumerate(res["cap_rows"]):
            w.writerow([i, *row])
    with open(out_dir / "report_examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nih_test_index", "true_findings", "cnn_probabilities", "dag_report", "monolith_report"])
        for i, row in enumerate(res["rep_rows"]):
            w.writerow([i, *row])
    with open(out_dir / "timings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rule", "level", "seconds", "finish_one_worker_per_rule_s", "on_critical_path"])
        for i, n in enumerate(res["names"]):
            w.writerow([n, res["levels"][i], f"{res['seconds'][n]:.2f}", f"{res['finish'][i]:.2f}",
                        int(n in res["path"])])  # fmt: skip
