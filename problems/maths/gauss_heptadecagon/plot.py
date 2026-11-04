"""Richmond's construction drawing itself, the Gauss period tree with its square roots, and the Gauss–Wantzel grid."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Polygon

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import product_in_periods

PER_STEP = 0.8  # frames per construction step (a step fades in over 1.25 frames)
POLY_FRAMES = 10
HOLD = 5
LABELLED = ["O", "P0", "B", "J", "E", "F", "K", "N3", "N5", "P3", "P4", "P5"]
FAINT = "#cbd5e1"


def _bitrev4(i: int) -> int:
    return int(f"{i:04b}"[::-1], 2)


def animate(d: dict, path: Path) -> Path:
    con, steps = d["con"], d["con"]["steps"]
    n_steps = len(steps)
    f_build = round(PER_STEP * n_steps)
    n_frames = f_build + POLY_FRAMES + HOLD

    fig = plt.figure(figsize=(10.4, 5.85))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.95], height_ratios=[1.25, 1.0])
    ax_c = fig.add_subplot(gs[:, 0])
    ax_t = fig.add_subplot(gs[0, 1])
    ax_g = fig.add_subplot(gs[1, 1])

    # ---------------------------------------------------------------- the construction
    lim = 1.32
    ax_c.set_xlim(-lim, lim)
    ax_c.set_ylim(-lim, lim)
    ax_c.set_aspect("equal")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.grid(False)
    artists = []
    for s in steps:
        if s["kind"] == "circle":
            c = s["centre"].double().cpu().numpy()
            art = Circle(c, float(s["radius"]), fill=False, lw=1.0, color=ACCENT[0], alpha=0)
            ax_c.add_patch(art)
        else:
            a, b = s["a"].double().cpu().numpy(), s["b"].double().cpu().numpy()
            u = (b - a) / np.linalg.norm(b - a)
            ends = np.stack([a - 4 * u, a + 4 * u])
            (art,) = ax_c.plot(ends[:, 0], ends[:, 1], color="#374151", lw=1.0, alpha=0)
        artists.append(art)
    pts = {k: v.double().cpu().numpy() for k, v in con["points"].items()}
    born = {name: i for i, s in enumerate(steps) for name in s["new"]}
    born.setdefault("O", -1)
    born.setdefault("P0", -1)
    dots = {name: ax_c.plot(*pts[name], "o", ms=3, color="#111827", alpha=0)[0] for name in born}
    tags = {
        name: ax_c.annotate(
            name.replace("P", "$P_{") + "}$" if name[0] == "P" else name,
            pts[name],
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0,
        )
        for name in LABELLED
    }
    verts = d["con"]["vertices"].double().cpu().numpy()
    vdots = ax_c.scatter(verts[:, 0], verts[:, 1], s=28, color=ACCENT[2], zorder=5, alpha=0)
    closed = np.vstack([verts, verts[:1]])
    (edges,) = ax_c.plot([], [], color=ACCENT[2], lw=2.2, zorder=4)
    fill = Polygon(verts, closed=True, color=ACCENT[2], alpha=0, lw=0, zorder=3)
    ax_c.add_patch(fill)
    title_c = ax_c.set_title("", fontsize=10)
    foot = ax_c.text(0.02, 0.02, "", transform=ax_c.transAxes, fontsize=8.5, color="#374151")

    # ---------------------------------------------------------------- the period tree and its square roots
    eta = d["tower"].cpu()
    order = d["order"].cpu()
    ax_t.set_xlim(-0.8, 15.8)
    ax_t.set_ylim(-9.6, 0.6)
    ax_t.axis("off")
    xs = {}
    for m in range(5):
        n = 2**m
        for i in range(n):
            leaves = [_bitrev4(j) for j in range(16) if j % n == i]
            xs[m, i] = float(np.mean(leaves))
    tree_art = {m: [] for m in range(5)}
    for m in range(5):
        for i in range(2**m):
            on_path = i == 0
            colour = ACCENT[1] if on_path else "#111827"
            if m < 4:
                label = f"{eta[m, i].real.item():.2f}"
            else:
                label = f"$\\zeta^{{{int(order[i])}}}$"
            size = 8 if m < 3 else 6.5
            t = ax_t.text(xs[m, i], -m * 1.15, label, ha="center", va="center", fontsize=size, color=colour, alpha=0)
            tree_art[m].append(t)
            if m:
                parent = i % 2 ** (m - 1)
                (ln,) = ax_t.plot(
                    [xs[m - 1, parent], xs[m, i]],
                    [-(m - 1) * 1.15 - 0.3, -m * 1.15 + 0.3],
                    color=ACCENT[1] if on_path else FAINT,
                    lw=1,
                    alpha=0,
                )
                tree_art[m].append(ln)
    for m, name in enumerate(["16 terms", "8 + 8", "4 × 4", "8 × 2: $2\\cos$", "16 × 1"]):
        ax_t.text(-0.7, -m * 1.15 - 0.45, name, fontsize=6.5, color="#6b7280")
    g = int(order[1])
    ax_t.set_title(
        rf"Gauss periods $\eta_{{m,i}} = \sum_j M_{{mij}}\,\zeta^{{{g}^j}}$, halved level by level", fontsize=10
    )
    lines = []
    for m in range(4):
        const, coef = product_in_periods(order, m)
        s = eta[m, 0]
        p = float(const[0]) + (coef[0].double() @ eta[m, : 2**m].real).item()
        r = eta[m + 1, 0]
        val = f"{r.real.item():.6f}" if m < 3 else f"{r.real.item():.4f} + {r.imag.item():.4f}i"
        lines.append(f"$t^2 - ({s.real.item():.4f})\\,t + ({p:.4f}) = 0$  →  $\\eta_{{{m + 1},0}}$ = {val}")
    radical = (
        r"$16\cos\frac{2\pi}{17} = -1+\sqrt{17}+\sqrt{34-2\sqrt{17}}"
        r"+2\sqrt{17+3\sqrt{17}-\sqrt{34-2\sqrt{17}}-2\sqrt{34+2\sqrt{17}}}$"
    )
    rad_art = [ax_t.text(-0.5, -6.0 - 0.75 * m, line, fontsize=8, alpha=0) for m, line in enumerate(lines)]
    rad_art.append(ax_t.text(-0.5, -9.1, radical, fontsize=8.5, alpha=0, color=ACCENT[1]))
    gauss_txt = ax_t.text(15.7, -9.55, "", fontsize=7.5, ha="right", va="top", color="#374151")

    # ---------------------------------------------------------------- Gauss–Wantzel grid
    grid = d["grid"]
    n_all = grid["n"].cpu().numpy()
    yes = grid["fermat"].cpu().numpy()
    n_max = len(n_all)
    cols = 10
    rows = -(-n_max // cols)
    code = np.where(yes, 2, 1)
    code[np.isin(n_all, [3, 5, 17, 257])] = 3
    code[n_all < 3] = 1
    full = np.zeros(rows * cols, dtype=int)
    cmap = ListedColormap(["#ffffff", "#f3f4f6", "#93c5fd", ACCENT[1]])
    img = np.zeros((rows, cols))
    im = ax_g.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation="nearest", aspect="auto")
    ax_g.set_xticks([])
    ax_g.set_yticks([])
    ax_g.grid(False)
    texts = []
    if rows <= 20:
        for idx, n in enumerate(n_all):
            texts.append(
                ax_g.text(
                    idx % cols,
                    idx // cols,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=7 if rows <= 10 else 5.5,
                    alpha=0,
                    color="#ffffff" if code[idx] == 3 else "#111827",
                    fontweight="bold" if code[idx] >= 2 else "normal",
                )
            )
    ax_g.set_title(
        f"Gauss–Wantzel: {int(yes.sum())} constructible n-gons for n ≤ {n_max} (red: Fermat primes)", fontsize=10
    )
    ax_g.set_xlabel(
        "$n = 2^k p_1\\cdots p_r$ with distinct Fermat primes $\\Leftrightarrow$ $\\varphi(n)$ a power of 2",
        fontsize=8.5,
    )

    err = d["err"].max().item()

    def update(f: int):
        u = min(f / PER_STEP, n_steps)  # construction progress in steps
        frac = min(1.0, f / f_build)
        for i, art in enumerate(artists):
            age = u - i
            if age <= 0:
                art.set_alpha(0)
                continue
            fresh = age < 2 and f < f_build
            art.set_alpha(min(1.0, age) * (1.0 if fresh else 0.25))
            art.set_linewidth(1.8 if fresh else 0.8)
            if steps[i]["kind"] == "circle":
                art.set_edgecolor(ACCENT[1] if fresh else ACCENT[0])
            else:
                art.set_color(ACCENT[1] if fresh else "#374151")
        for name, dot in dots.items():
            dot.set_alpha(0.9 if u > born[name] + 1 else 0)
        for name, tag in tags.items():
            tag.set_alpha(1 if u > born[name] + 1 else 0)
        shown = [k for k in range(17) if u > born.get(f"P{k}", -1) + 1]
        vdots.set_alpha(1 if shown else 0)
        vdots.set_offsets(verts[shown] if shown else np.empty((0, 2)))
        k_step = min(n_steps - 1, int(u))
        if f < f_build:
            title_c.set_text(f"step {k_step + 1}/{n_steps}: {steps[k_step]['label']}")
        else:
            g = (f - f_build) / POLY_FRAMES
            m = min(17, round(17 * g)) + 1
            edges.set_data(closed[:m, 0], closed[:m, 1])
            fill.set_alpha(0.25 * min(1.0, max(0.0, 2 * g - 1)))
            title_c.set_text("the regular 17-gon, Richmond (1893)")
            foot.set_text(f"{n_steps} ruler-and-compass steps, max vertex error {err:.1e} ({d['precision']})")
        for m in range(5):
            for art in tree_art[m]:
                art.set_alpha(1 if frac >= m / 5 else 0)
        for m, art in enumerate(rad_art):
            art.set_alpha(1 if frac >= (m + 1) / 5 else 0)
        if frac >= 1:
            gauss_txt.set_text(f"= {16 * d['gauss']:.13f}, cos error {abs(d['gauss'] - np.cos(2 * np.pi / 17)):.0e}")
        reveal = int(round(n_max * min(1.0, f / (0.8 * f_build))))
        full[:] = 0
        full[:reveal] = code[:reveal]
        im.set_data(full.reshape(rows, cols))
        for idx, t in enumerate(texts):
            t.set_alpha(1 if idx < reveal else 0)

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(n_frames, n_frames), path, fps=10)
