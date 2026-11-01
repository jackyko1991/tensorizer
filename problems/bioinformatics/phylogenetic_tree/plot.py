"""Animate tree reconstruction as more sites are used: true tree, distances, NJ and UPGMA dendrograms."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif

from .solve import clades, splits


def layout(tree):
    """x = distance from the root, y = leaf order (depth-first) and the mean of the children's y for inner nodes."""
    kids = tree.children()
    x, y = [0.0] * len(tree.parent), [0.0] * len(tree.parent)
    order: list[int] = []

    def visit(v: int, x0: float):
        x[v] = x0
        if not kids[v]:
            y[v] = len(order)
            order.append(v)
            return
        for c in kids[v]:
            visit(c, x0 + tree.length[c])
        y[v] = float(np.mean([y[c] for c in kids[v]]))

    visit(tree.root, 0.0)
    return x, y, kids


def draw(ax, tree, true_splits: set, title: str):
    """Rectangular dendrogram; branches whose split is not in the true tree are red."""
    ax.clear()
    x, y, kids = layout(tree)
    leaves = clades(tree)
    n = tree.n_leaves

    def wrong(v: int) -> bool:  # the branch above v makes a split that the true tree does not have
        side = leaves[v] if 0 not in leaves[v] else frozenset(range(n)) - leaves[v]
        return 2 <= len(side) <= n - 2 and side not in true_splits

    for v, cs in enumerate(kids):
        style = {"color": "#dc2626", "lw": 2.2} if wrong(v) else {"color": "#374151", "lw": 1.4}
        if cs:
            ax.plot([x[v], x[v]], [min(y[c] for c in cs), max(y[c] for c in cs)], **style)
        if tree.parent[v] >= 0:
            ax.plot([x[tree.parent[v]], x[v]], [y[v], y[v]], **style)
    span = max(x) or 1.0
    for leaf in range(n):
        ax.plot(x[leaf], y[leaf], "o", color="#2563eb", ms=4)
        ax.text(x[leaf] + 0.03 * span, y[leaf], f"t{leaf}", va="center", fontsize=8)
    ax.set_xlim(-0.03 * span, 1.18 * span)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("substitutions per site", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)


def animate(true_tree, results, d_true, out: Path) -> Path:
    """results: list of (sites, p, d, nj_tree, upgma_tree, rf_nj, rf_upgma); d_true (n, n) path lengths."""
    true_splits = splits(true_tree)
    iu = np.triu_indices(true_tree.n_leaves, 1)
    dt = d_true.cpu().numpy()[iu]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    ax_true, ax_d, ax_nj, ax_up = axes.ravel()
    draw(ax_true, true_tree, true_splits, "true tree (sequences evolved along it)")
    top = 1.15 * dt.max()
    ax_d.plot([0, top], [0, top], color="#9ca3af", ls="--", lw=1)
    sc_p = ax_d.scatter(dt, dt, s=12, color="#9ca3af", label="p-distance")
    sc_d = ax_d.scatter(dt, dt, s=12, color="#2563eb", label="Jukes–Cantor $d$")
    ax_d.set_xlim(0, top)
    ax_d.set_ylim(0, top)
    ax_d.set_xlabel("true path length on the tree", fontsize=8)
    ax_d.set_ylabel("estimated distance", fontsize=8)
    ax_d.legend(fontsize=7, loc="upper left")
    ax_d.tick_params(labelsize=7)
    head = ax_d.set_title(" ", fontsize=9)

    def update(k: int):
        sites, p, d, nj, up, rf_nj, rf_up = results[k]
        sc_p.set_offsets(np.c_[dt, p.cpu().numpy()[iu]])
        sc_d.set_offsets(np.c_[dt, d.cpu().numpy()[iu]])
        head.set_text(f"pairwise distances from {sites} sites")
        draw(ax_nj, nj, true_splits, f"neighbour joining: RF = {rf_nj}")
        draw(ax_up, up, true_splits, f"UPGMA: RF = {rf_up}  (red: wrong split)")

    update(0)
    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(results), 60), out, fps=4)
