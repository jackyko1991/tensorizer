"""p(n) against Hardy–Ramanujan and Rademacher as n grows, and Ramanujan's congruences lighting up on residue grids."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from problems._common.plotting import ACCENT, frame_indices, save_gif

N_FRAMES = 40


def animate(d: dict, path: Path) -> Path:
    n = d["n"].cpu().numpy()
    ratio = d["ratio"].cpu().numpy()
    n_rad = d["n_rad"].cpu().numpy()
    err = np.maximum(d["err"].cpu().numpy(), 1e-3)
    k_show = sorted({1, 2, 3, d["n_terms"]})
    p_mod = d["p_mod"].cpu().numpy()
    cong = d["congruences"].cpu().numpy()[:3]  # mod 5, 7, 11
    grid_n = d["grid_n"]

    fig = plt.figure(figsize=(12, 6.8))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1])
    ax_r = fig.add_subplot(gs[0, :3])
    ax_e = fig.add_subplot(gs[0, 3:])
    ax_g = [fig.add_subplot(gs[1, 2 * i : 2 * i + 2]) for i in range(3)]

    # Hardy–Ramanujan ratio
    (ln_r,) = ax_r.semilogx(
        n[:1], ratio[:1], color=ACCENT[0], lw=1.8, label=r"$p(n)\,/\,\frac{e^{\pi\sqrt{2n/3}}}{4n\sqrt{3}}$"
    )
    ax_r.semilogx(n, 1 - d["c1"] / np.sqrt(n), color=ACCENT[1], lw=1, ls="--", label=r"$1 - c/\sqrt{n}$")
    ax_r.axhline(1, color="#9ca3af", lw=1)
    ax_r.set_xlim(1, n[-1])
    ax_r.set_ylim(0.55, 1.03)
    ax_r.set_xlabel("$n$")
    ax_r.set_ylabel("ratio")
    ax_r.legend(fontsize=9, loc="lower right")
    title_r = ax_r.set_title("", fontsize=10)

    # Rademacher partial sums
    lines_e = []
    for i, k in enumerate(k_show):
        (ln,) = ax_e.semilogy(
            n_rad[:1], err[:1, k - 1], color=ACCENT[i], lw=1.3, label=f"{k} term{'s' if k > 1 else ''}"
        )
        lines_e.append((ln, k))
    ax_e.axhline(0.5, color="#111827", lw=1, ls=":", label="0.5: rounding is exact below")
    ax_e.set_xlim(1, n_rad[-1])
    ax_e.set_ylim(1e-3, err[:, 0].max() * 2)
    ax_e.set_xlabel("$n$")
    ax_e.set_ylabel("|Rademacher − p(n)|")
    ax_e.legend(fontsize=8, loc="upper left", ncol=2)
    title_e = ax_e.set_title("Rademacher's exact series, partial sums", fontsize=10)

    # residue grids: row r = n mod m, column k = n // m; 0 not drawn, 1 nonzero, 2 zero, 3 zero on Ramanujan's row
    cmap = ListedColormap(["#ffffff", "#e5e7eb", "#93c5fd", "#dc2626"])
    imgs, codes = [], []
    for ax, (a, b, m) in zip(ax_g, cong):
        cols = -(-grid_n // m)
        nn = np.arange(cols * m)
        res = np.where(nn < grid_n, p_mod[np.minimum(nn, len(p_mod) - 1), list(cong[:, 2]).index(m)], -1)
        code = np.where(res == 0, np.where(nn % m == b, 3, 2), 1)
        code = code.reshape(cols, m).T  # (m, cols)
        img = np.zeros_like(code)
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest", origin="lower")
        ax.set_yticks(range(m))
        ax.set_yticklabels([str(r) if r in (0, b, m - 1) else "" for r in range(m)], fontsize=7)
        ax.grid(False)
        ax.set_xlabel(f"$k$ = n // {m}", fontsize=8)
        ax.set_ylabel(f"n mod {m}", fontsize=8)
        ax.set_title(f"$p(n)$ mod {m}: red row $p({a}k+{b})\\equiv 0$", fontsize=9)
        imgs.append(im)
        codes.append((img, code))

    fig.text(
        0.5,
        0.01,
        "Ramanujan noticed that… p(5k+4), p(7k+5) and p(11k+6) are divisible by 5, 7 and 11"
        " (blue: other zeros, which come and go)",
        ha="center",
        fontsize=9,
        color="#6b7280",
        style="italic",
    )

    def update(f: int):
        s = (f + 1) / N_FRAMES
        j = max(2, int(round(len(n) ** s)))  # n grows geometrically to match the log axis
        ln_r.set_data(n[:j], ratio[:j])
        title_r.set_text(f"Hardy–Ramanujan: ratio → 1   (n = {n[j - 1]:,}: {ratio[j - 1]:.4f})")
        i = max(2, round(len(n_rad) * s))
        for ln, k in lines_e:
            ln.set_data(n_rad[:i], err[:i, k - 1])
        title_e.set_text(f"Rademacher's exact series, partial sums (n ≤ {n_rad[i - 1]})")
        shown = max(1, round(grid_n * s))
        for im, (img, code), (_, _, m) in zip(imgs, codes, cong):
            flat = np.zeros(code.size, dtype=int)
            full = code.T.flatten()
            flat[:shown] = full[:shown]
            img[:] = flat.reshape(-1, m).T
            im.set_data(img)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return save_gif(fig, update, frame_indices(N_FRAMES, N_FRAMES), path, fps=6)
