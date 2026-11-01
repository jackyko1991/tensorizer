"""Animate the probabilities flowing over the graph, the ranks from three methods, and the convergence rates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(links, community, pos, history, pi_eig, occupancy, residual, ds, factor, lam2, d, path: Path) -> Path:
    """history (K+1, N) power iterates for damping d, occupancy (K+1, N) surfer fractions, residual (K, B) for ds."""
    lk, xy = links.cpu().numpy(), pos.cpu().numpy()
    x, mc, pi = history.cpu().numpy(), occupancy.cpu().numpy(), pi_eig.cpu().numpy()
    res, rate, lam = residual.cpu().numpy(), factor.cpu().numpy(), lam2.cpu().numpy()
    n, n_steps = len(pi), len(x)
    colours = np.array([ACCENT[c] for c in community.tolist()])
    dangling = lk.sum(axis=0) == 0

    fig, (ax_g, ax_b, ax_r) = plt.subplots(1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    for i, j in zip(*np.nonzero(lk)):  # an arrow for every link j → i
        ax_g.annotate(
            "",
            xy=xy[i],
            xytext=xy[j],
            arrowprops={"arrowstyle": "-|>", "color": "#9ca3af", "lw": 0.6, "shrinkA": 4, "shrinkB": 6},
        )
    nodes = ax_g.scatter(xy[:, 0], xy[:, 1], s=40, c=colours, zorder=3, edgecolors="#374151", linewidths=0.6)
    ax_g.scatter(*xy[dangling].T, s=90, facecolors="none", edgecolors="#dc2626", linewidths=1.5, zorder=4)
    ax_g.set_aspect("equal")
    ax_g.axis("off")
    ax_g.set_title("web graph (red rings: dangling pages)", fontsize=9)

    order = np.argsort(-pi)  # pages sorted by their final rank
    ranks = np.arange(n)
    bars = ax_b.bar(ranks, x[0, order], color=colours[order], alpha=0.75, label="power iteration $x_k$")
    (dots,) = ax_b.plot(ranks, mc[0, order], "o", color="#f59e0b", ms=4, label="surfer fractions")
    ax_b.plot(ranks, pi[order], "_", color="#111827", ms=9, mew=1.5, label="eigenvector of $G$")
    ax_b.set_xlabel("page, sorted by PageRank")
    ax_b.set_ylabel("probability")
    ax_b.set_ylim(0, 1.25 * max(pi.max(), mc.max()))
    ax_b.legend(fontsize=7, loc="upper right")

    k = np.arange(1, res.shape[0] + 1)
    for b, db in enumerate(ds):
        col = ACCENT[b % len(ACCENT)]
        ax_r.semilogy(
            k,
            res[:, b],
            color=col,
            lw=1.6 if db == d else 0.9,
            label=f"d = {db:.2f}: rate {rate[b]:.3f}, |λ₂| {lam[b]:.3f}",
        )
        ax_r.semilogy(k, res[0, b] * db ** (k - 1.0), "--", color=col, lw=0.7, alpha=0.6)
    ax_r.set_ylim(1e-15, 1)
    ax_r.set_xlabel("iteration $k$")
    ax_r.set_title(r"$\|x_{k+1} - x_k\|_1$ (dashed: $d^k$)", fontsize=9)
    ax_r.legend(fontsize=7, loc="lower left")
    marker = ax_r.axvline(0, color="#374151", lw=0.8)
    title = fig.suptitle("", fontsize=11)

    def update(t: int):
        nodes.set_sizes(25 + 2500 * x[t])
        for bar, h in zip(bars, x[t, order]):
            bar.set_height(h)
        dots.set_ydata(mc[t, order])
        marker.set_xdata([max(t, 1)])
        title.set_text(f"PageRank with damping d = {d}: step {t}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(n_steps, 120), path, fps=6)
