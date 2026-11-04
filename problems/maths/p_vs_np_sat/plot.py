"""Solution map of one formula as clauses are added, P(sat) against α, search cost, and verify vs search time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(alpha, p_sat, maps, record, nodes, cost, n, threshold, path: Path) -> Path:
    fig, ((ax_m, ax_p), (ax_w, ax_t)) = plt.subplots(2, 2, figsize=(10, 6.6))
    image = maps.numpy().reshape(len(maps), 2 ** (n // 2), -1)  # (M + 1, 2^(n/2), 2^(n − n/2)): one cell each
    im = ax_m.imshow(image[0], cmap=ListedColormap(["#f3f4f6", "#2563eb"]), vmin=0, vmax=1, interpolation="nearest")
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    ax_m.grid(False)
    title_m = ax_m.set_title("", fontsize=9)
    ax_m.set_autoscale_on(False)
    few = ax_m.scatter([], [], s=14, color="#2563eb", edgecolors="#1e40af")  # rare survivors drawn larger

    for (size, p), colour in zip(p_sat.items(), ACCENT):
        ax_p.plot(alpha[size].numpy(), p.numpy(), color=colour, lw=1.5, label=f"$n = {size}$")
    ax_p.axvline(threshold, color="#9ca3af", ls="--", lw=1)
    ax_p.text(threshold + 0.08, 0.92, r"$\alpha_c \approx 4.27$", fontsize=8, color="#6b7280")
    ax_p.set_xlabel(r"clauses per variable $\alpha = M/n$")
    ax_p.set_ylabel("fraction satisfiable")
    ax_p.set_title("satisfiability phase transition", fontsize=9)
    ax_p.legend(fontsize=8, loc="lower left")
    cursor_p = ax_p.axvline(0, color="#dc2626", lw=1)

    a_rec = np.array(record) / n
    ax_w.semilogy(a_rec, nodes.numpy(), "o-", color=ACCENT[2], ms=3, lw=1.3)
    ax_w.axvline(threshold, color="#9ca3af", ls="--", lw=1)
    ax_w.set_xlabel(r"$\alpha = M/n$")
    ax_w.set_ylabel("backtracking nodes (median)")
    ax_w.set_title(f"search cost at $n = {n}$: easy – hard – easy", fontsize=9)
    cursor_w = ax_w.axvline(0, color="#dc2626", lw=1)

    sizes = np.array(cost["n"])
    ax_t.semilogy(sizes, np.array(cost["search"]) * 1e3, "o-", color=ACCENT[1], label="search all $2^n$ assignments")
    ax_t.semilogy(sizes, np.array(cost["verify"]) * 1e3, "o-", color=ACCENT[0], label="verify a certificate")
    ax_t.set_xlabel("number of variables $n$")
    ax_t.set_ylabel("time for 64 formulas [ms]")
    ax_t.set_title(r"checking is cheap, exhaustive search doubles per variable", fontsize=9)
    ax_t.legend(fontsize=8, loc="upper left")

    def update(m: int):
        im.set_data(image[m])
        a = m / n
        k = int(image[m].sum())
        rows, cols = np.nonzero(image[m])
        few.set_offsets(np.c_[cols, rows] if k <= 200 else np.empty((0, 2)))
        title_m.set_text(f"one formula, M = {m} clauses (α = {a:.2f}): {k:,} of {2**n:,} assignments satisfy it")
        cursor_p.set_xdata([a, a])
        cursor_w.set_xdata([a, a])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(image), 48), path, fps=5)
