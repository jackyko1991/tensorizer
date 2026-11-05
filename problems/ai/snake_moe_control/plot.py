"""Animate the result: six games played at once by a routed controller (each board's frame in the colour of the
expert the router chose at that step), the router weights of every game, the router of game 1 over time, the
imitation learning curves and the score of every policy with one dot per seed. Also a static figure of what the
experts are used for (routing vs state features) and the CSV tables."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from problems._common.plotting import ACCENT, frame_indices, save_png
from problems.ai.chain_of_thought.plot import save_gif_shared_palette

from . import tensors as T

EXPERT_COLORS = ["#ea580c", "#16a34a", "#9333ea", "#0891b2", "#db2777", "#65a30d", "#a16207", "#4b5563"]
POLICY_COLORS = {"random": "#9ca3af", "dense": ACCENT[0], "switch": ACCENT[1], "dense_top1": ACCENT[3],
                 "mono": ACCENT[2], "teacher": "#374151"}  # fmt: skip
POLICIES = ("random", "dense", "switch", "dense_top1", "mono", "teacher")
SHORT = {"random": "random", "dense": "(a)\ndense", "switch": "(b)\nSwitch", "dense_top1": "(c)\ndense→top-1",
         "mono": "(d)\nmono", "teacher": "teacher"}  # fmt: skip
FEATURES = {"dist": ("distance to food (cells)", ["≤1", "2–3", "4–6", "≥7"]),
            "ahead": ("free cells ahead (cells)", ["0", "1", "2–3", "≥4"]),
            "length": ("body length (cells)", ["≤4", "5–8", "9–14", "≥15"]),
            "turn": ("controller's move (dimensionless)", ["left", "straight", "right"])}  # fmt: skip


def shown_variant(res: dict) -> str:
    """(c) plays in the GIF: every game runs exactly one expert, and its router uses all of them (the Switch
    router of (b) tends to collapse onto one or two experts; routing.png shows all three routed variants)."""
    return "dense_top1"


def board_rgb(age: np.ndarray, length: int, food: np.ndarray) -> np.ndarray:
    """N × N board → RGB: light grey cells, the body from dark (head) to light blue (tail), the food red."""
    n = age.shape[0]
    img = np.ones((n, n, 3)) * np.array([0.95, 0.96, 0.97])
    frac = age / max(length, 1)
    dark, light = np.array([0.09, 0.25, 0.69]), np.array([0.66, 0.80, 0.98])
    body = age > 0
    img[body] = light + (dark - light) * frac[body][:, None]
    img[food[0], food[1]] = [0.86, 0.15, 0.15]
    return img


def animate(res: dict, path: Path, n_frames: int = 80) -> Path:
    cfg, S, K = res["cfg"], res["cfg"].seeds, res["cfg"].experts
    v = shown_variant(res)
    mi = T.VARIANTS.index(v) * S  # seed 1 of that variant
    fr = res["frames"]
    n_t = min(cfg.show_steps, fr["age"].shape[0])
    show = fr["age"].shape[2]
    frames = frame_indices(n_t, n_frames)
    age, food, alive = fr["age"][:, mi].numpy(), fr["food"][:, mi].numpy(), fr["alive"][:, mi].numpy()
    length, p = fr["length"][:, mi].numpy(), fr["p"][:, mi].numpy()  # (T, show), (T, show, K)

    fig = plt.figure(figsize=(14.5, 9.4))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1.05], hspace=0.55, wspace=0.6,
                  left=0.05, right=0.985, top=0.93, bottom=0.08)  # fmt: skip
    boards = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(min(show, 6))]
    ax_bar, ax_time = fig.add_subplot(gs[0, 3:]), fig.add_subplot(gs[1, 3:])
    ax_learn, ax_score = fig.add_subplot(gs[2, :3]), fig.add_subplot(gs[2, 3:])
    fig.suptitle(f"Snake, 6 of {cfg.eval_games} games batched, controller {T.LABELS[v]} (seed 1): "
                 "frame colour = routed expert", fontsize=13)  # fmt: skip

    ims = []
    for g, ax in enumerate(boards):
        ims.append(ax.imshow(board_rgb(age[0, g], length[0, g], food[0, g]), interpolation="nearest"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if g >= 3:
            ax.set_xlabel("column (cells)", fontsize=10)
        if g % 3 == 0:
            ax.set_ylabel("row (cells)", fontsize=10)

    # router weights of the six games now
    lefts = np.zeros(len(boards))
    bars = []
    for k in range(K):
        bars.append(ax_bar.barh(np.arange(len(boards)), p[0, : len(boards), k], left=lefts, color=EXPERT_COLORS[k],
                                label=f"expert {k + 1}", height=0.7))  # fmt: skip
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(len(boards) - 0.5, -0.5)
    ax_bar.set_yticks(np.arange(len(boards)), [f"game {g + 1}" for g in range(len(boards))])
    ax_bar.set_xlabel("router weight $p_k$ (dimensionless)")
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=K, fontsize=9, frameon=False)

    # router of game 1 over time
    t_axis = np.arange(n_t)
    ax_time.set_xlim(0, n_t - 1)
    ax_time.set_ylim(0, 1)
    ax_time.set_xlabel("game step $t$ (moves)")
    ax_time.set_ylabel("$p_k$ (dimensionless)")
    ax_time.set_title("router of game 1 over time (stacked $p_k$)")
    cursor = ax_time.axvline(0, color="#111827", lw=1.2)

    # learning curves (static)
    loss = res["loss"].numpy()
    steps = np.arange(1, len(loss) + 1)
    for j, name in enumerate(("dense", "switch", "mono")):
        c = POLICY_COLORS[name]
        for s in range(S):
            ax_learn.plot(steps, _smooth(loss[:, j * S + s]), color=c, lw=1.0, alpha=0.6,
                          label=T.LABELS[name] if s == 0 else None)  # fmt: skip
    if res["dagger_at"]:
        ax_learn.axvline(res["dagger_at"], color="#9ca3af", ls="--", lw=1)
        ax_learn.text(res["dagger_at"], 0.97, "DAgger data added ", transform=ax_learn.get_xaxis_transform(),
                      va="top", ha="right", fontsize=9, color="#4b5563")  # fmt: skip
    ax_learn.set_yscale("log")
    ax_learn.set_xlim(1, len(loss))
    lo, hi = np.nanmin(_smooth(loss[:, 0])), np.nanmax(loss)
    ax_learn.set_ylim(min(lo, np.nanmin(loss[len(loss) // 2 :])) * 0.8, hi * 1.2)
    ax_learn.set_xlabel("training step (minibatches)")
    ax_learn.set_ylabel("imitation loss (nats)")
    ax_learn.set_title("learning: cross-entropy vs the teacher (one line per seed)")
    ax_learn.legend(fontsize=9, loc="upper right")

    # scores (static)
    _score_bars(ax_score, res)

    state = {"fill": []}

    def update(f):
        for g, ax in enumerate(boards):
            ims[g].set_data(board_rgb(age[f, g], length[f, g], food[f, g]))
            k = int(p[f, g].argmax())
            col = EXPERT_COLORS[k] if alive[f, g] else "#9ca3af"
            for sp in ax.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(5)
            ax.set_title(f"game {g + 1}: length {length[f, g]}" + ("" if alive[f, g] else " (ended)"),
                         fontsize=11, color="#111827" if alive[f, g] else "#6b7280")  # fmt: skip
        lefts = np.zeros(len(boards))
        for k in range(K):
            for g, rect in enumerate(bars[k]):
                w = p[f, g, k] if alive[f, g] else 0.0
                rect.set_x(lefts[g])
                rect.set_width(w)
                lefts[g] += w
        ax_bar.set_title(f"router weights now, step $t$ = {f}")
        for c in state["fill"]:
            c.remove()
        upto = int(f) + 1
        live = alive[:upto, 0]
        pp = np.where(live[:, None], p[:upto, 0], np.nan)
        state["fill"] = list(ax_time.stackplot(t_axis[:upto], pp.T, colors=EXPERT_COLORS[:K], alpha=0.9))
        cursor.set_xdata([f, f])

    return save_gif_shared_palette(fig, update, frames, path, fps=6)


def _smooth(y: np.ndarray, k: int = 25) -> np.ndarray:
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def _score_bars(ax, res: dict) -> None:
    hi = max(float(res["food"][v].mean(-1).max()) for v in POLICIES)
    for i, v in enumerate(POLICIES):
        x = res["food"][v]
        per_seed = x.mean(-1).numpy() if x.dim() > 1 else np.array([float(x.mean())])
        ax.bar(i, per_seed.mean(), color=POLICY_COLORS[v], alpha=0.85, width=0.7)
        ax.scatter(np.full(len(per_seed), i) + np.linspace(-0.12, 0.12, len(per_seed)), per_seed, color="#111827",
                   s=18, zorder=3, label="one seed" if i == 1 else None)  # fmt: skip
        ax.text(i, per_seed.mean() + 0.02 * hi, f"{per_seed.mean():.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(POLICIES)), [SHORT[v] for v in POLICIES], fontsize=10)
    ax.set_ylim(0, hi * 1.18 + 0.5)
    ax.set_xlim(-0.6, len(POLICIES) - 0.4)
    ax.set_ylabel("food eaten per game (count)")
    cfg = res["cfg"]
    ax.set_title(f"score: {cfg.eval_games} shared games, ≤ {cfg.eval_steps} moves")
    ax.legend(fontsize=9, loc="upper left")


def routing_png(res: dict, path: Path) -> Path:
    """P(expert | state feature bin) for each routed variant (seed 1), with the mutual information of every seed."""
    S, K = res["cfg"].seeds, res["cfg"].experts
    sp = res["spec"]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), squeeze=False)
    for r, v in enumerate(("dense", "switch", "dense_top1")):
        m = r * S
        for c, (feat, (xlabel, ticks)) in enumerate(FEATURES.items()):
            ax = axes[r, c]
            tab = sp["table"][feat][m].numpy()  # (bins, K)
            bottom = np.zeros(len(tab))
            for k in range(K):
                ax.bar(np.arange(len(tab)), tab[:, k], bottom=bottom, color=EXPERT_COLORS[k], width=0.75,
                       label=f"expert {k + 1}")  # fmt: skip
                bottom += tab[:, k]
            ax.set_xticks(range(len(ticks)), ticks)
            ax.set_ylim(0, 1)
            ax.set_xlabel(xlabel)
            mis = ", ".join(f"{float(sp['mi'][feat][r * S + s]):.2f}" for s in range(S))
            ax.set_title(f"{T.LABELS[v]}: MI {mis} bit", fontsize=11)
            if c == 0:
                ax.set_ylabel("share routed (dimensionless)")
    axes[0, 3].legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("What the router uses: share of states sent to each expert per feature bin (seed 1); "
                 "MI = mutual information, one value per seed", fontsize=13)  # fmt: skip
    fig.tight_layout()
    return save_png(fig, path)


def write_tables(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    S = res["cfg"].seeds
    with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "seed", "games", "food_mean", "food_std", "moves_mean"])
        for v in POLICIES:
            x, st = res["food"][v], res["steps"][v]
            rows = [(0, x, st)] if x.dim() == 1 else [(s + 1, x[s], st[s]) for s in range(S)]
            for s, xs, ss in rows:
                w.writerow([v, s or "-", len(xs), f"{float(xs.mean()):.3f}", f"{float(xs.std()):.3f}",
                            f"{float(ss.mean()):.1f}"])  # fmt: skip
    sp = res["spec"]
    with (out_dir / "routing.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "feature", "bin", "expert", "share_routed", "mutual_information_bit"])
        for r, v in enumerate(("dense", "switch", "dense_top1")):
            for s in range(S):
                m = r * S + s
                for k in range(res["cfg"].experts):
                    w.writerow([v, s + 1, "all", "all", k + 1, f"{float(sp['load'][m, k]):.4f}", ""])
                for feat, (_, ticks) in FEATURES.items():
                    for b, tick in enumerate(ticks):
                        for k in range(res["cfg"].experts):
                            w.writerow([v, s + 1, feat, tick, k + 1, f"{float(sp['table'][feat][m, b, k]):.4f}",
                                        f"{float(sp['mi'][feat][m]):.4f}"])  # fmt: skip
