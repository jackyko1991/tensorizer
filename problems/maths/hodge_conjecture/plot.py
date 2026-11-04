"""Hodge diamonds one by one, with Picard numbers of Fermat surfaces, Hodge classes of Fermat fourfolds and χ checks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(shown: list, surf: dict, four: dict, check, path: Path) -> Path:
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1])
    ax_h = fig.add_subplot(grid[:, 0])
    ax_s = fig.add_subplot(grid[0, 1:])
    ax_f = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[1, 2])

    ds = np.array(surf["d"])
    w = 0.27
    ax_s.bar(ds - w, surf["h11"], w, color="#cbd5e1", label=r"$h^{1,1}$")
    ax_s.bar(ds, surf["rho"], w, color=ACCENT[0], label=r"Hodge classes $\rho$ (Shioda) = algebraic (Lefschetz)")
    ax_s.bar(ds + w, surf["lines"], w, color=ACCENT[4], label=r"rank spanned by the $3d^2$ lines")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("degree $d$ of the Fermat surface $x^d + y^d + z^d + w^d = 0$")
    ax_s.legend(fontsize=7, loc="upper left")
    ax_s.set_title(r"surfaces: Hodge classes in $H^2$ are all algebraic (proved)", fontsize=9)
    mark_s = ax_s.axvspan(-11, -10, color="#fde68a", alpha=0.5, zorder=0)

    d4 = np.array(four["d"])
    ax_f.bar(d4 - 0.2, four["h22"], 0.4, color="#cbd5e1", label=r"$h^{2,2}$")
    ax_f.bar(d4 + 0.2, four["hodge"], 0.4, color=ACCENT[3], label="rational Hodge classes")
    ax_f.set_yscale("log")
    ax_f.set_xlabel("degree $d$ (Fermat fourfold)")
    ax_f.legend(fontsize=7, loc="upper left")
    ax_f.set_title("fourfolds: algebraic? (the conjecture)", fontsize=9)
    mark_f = ax_f.axvspan(-11, -10, color="#fde68a", alpha=0.5, zorder=0)

    chk = check.numpy()
    ax_e.scatter(np.abs(chk[:, 3]) + 1, np.abs(chk[:, 2]) + 1, s=14, c=chk[:, 0], cmap="viridis", zorder=3)
    top = np.abs(chk[:, 2:]).max() + 1
    ax_e.plot([1, top], [1, top], color=ACCENT[1], lw=1)
    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlabel(r"$|\chi|+1$ from $((1-d)^{n+2}-1)/d + n + 2$")
    ax_e.set_ylabel(r"$|\chi|+1$ from the diamond")
    ax_e.set_title(f"Euler characteristic, {len(chk)} hypersurfaces", fontsize=9)
    (mark_e,) = ax_e.plot([], [], "o", ms=11, mfc="none", mec=ACCENT[1], mew=2)

    def draw_diamond(case: dict):
        ax_h.clear()
        h = case["h"].numpy()
        n = h.shape[0] - 1
        vmax = np.log1p(h.max())
        for p in range(n + 1):
            for q in range(n + 1):
                x, y = q - p, p + q  # h^{0,0} at the bottom, h^{n,n} at the top
                middle = p + q == n
                colour = plt.cm.Blues(0.15 + 0.6 * np.log1p(h[p, q]) / max(vmax, 1e-9))
                ax_h.scatter(
                    x,
                    y,
                    s=1300 if n <= 4 else 750,
                    marker="D",
                    color=colour,
                    edgecolor=ACCENT[1] if middle else "#9ca3af",
                    lw=1.6 if middle else 0.8,
                )
                size = 11 if h[p, q] < 1000 else 8
                ax_h.text(
                    x,
                    y,
                    str(h[p, q]),
                    ha="center",
                    va="center",
                    fontsize=size if n <= 4 else size - 2,
                    color="white" if np.log1p(h[p, q]) > 0.6 * vmax and h[p, q] > 1 else "#111827",
                )
        ax_h.set_xlim(-n - 1, n + 1)
        ax_h.set_ylim(-1.2, 2 * n + 1.5)
        ax_h.set_aspect("equal")
        ax_h.axis("off")
        lines = [case["label"], f"dimension n = {n},  χ = {case['chi']}"]
        if "chi_formula" in case:
            lines[-1] += f"  (formula: {case['chi_formula']})"
        if case.get("hodge") is not None:
            lines.append(
                rf"rational Hodge classes in $H^{{{n}}}$: {case['hodge']} "
                rf"of $h^{{{n // 2},{n // 2}}}$ = {h[n // 2, n // 2]}"
            )
        if "torus" in case:
            lines.append(r"$h^{p,q} = \binom{g}{p}\binom{g}{q}$")
        ax_h.set_title("\n".join(lines), fontsize=9)
        ax_h.text(
            0,
            -1.0,
            r"symmetric under $p \leftrightarrow q$ (Hodge) and $(p,q) \to (n-p, n-q)$ (Serre)",
            ha="center",
            fontsize=7.5,
            color="#6b7280",
        )

    def update(f: int):
        case = shown[f]
        draw_diamond(case)
        fermat = case.get("fermat")
        for patch, xs, on in ((mark_s, ds, fermat and fermat[0] == 2), (mark_f, d4, fermat and fermat[0] == 4)):
            x = fermat[1] if on and fermat[1] in xs else -10
            patch.set_x(x - 0.5)
            patch.set_width(1.0)
        if fermat:
            row = chk[(chk[:, 0] == fermat[0]) & (chk[:, 1] == fermat[1])]
            mark_e.set_data(np.abs(row[:, 3]) + 1, np.abs(row[:, 2]) + 1)
        else:
            mark_e.set_data([], [])

    ax_s.set_xlim(ds[0] - 0.6, ds[-1] + 0.6)
    ax_f.set_xlim(d4[0] - 0.6, d4[-1] + 0.6)
    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(shown), len(shown)), path, fps=1)
