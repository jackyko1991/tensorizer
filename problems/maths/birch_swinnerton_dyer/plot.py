"""Partial products against log log X, a_p inside the Hasse band, the Sato–Tate histogram and L(E, s) near s = 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif


def _slope(loglog: np.ndarray, v: np.ndarray) -> float:
    uc = loglog - loglog.mean()
    return float(((v - v.mean()) * uc).sum() / (uc * uc).sum())


def animate(names, ranks, primes, x, log_prod, lf, frames, path: Path) -> Path:
    p = primes.numpy().astype(float)
    x, log_prod = x.numpy(), log_prod.numpy()
    loglog = np.log(np.log(p))
    fig, ((ax_g, ax_a), (ax_h, ax_l)) = plt.subplots(2, 2, figsize=(10, 6.6))

    lines = [
        ax_g.plot([], [], color=ACCENT[c], lw=1.4, label=f"{name}, rank {r}")[0]
        for c, (name, r) in enumerate(zip(names, ranks))
    ]
    ax_g.set_xlim(loglog[0], loglog[-1] + 0.05)
    ax_g.set_ylim(min(0.0, log_prod.min()) - 0.3, log_prod.max() + 0.5)
    ax_g.set_xlabel(r"$\log\log X$")
    ax_g.set_ylabel(r"$\log \prod_{p \leq X} N_p / p$")
    legend = ax_g.legend(fontsize=7, loc="upper left")
    title_g = ax_g.set_title("", fontsize=9)

    pick = 1  # 37a1
    dots = ax_a.scatter([], [], s=2, color=ACCENT[pick])
    for level in (-1, 1):
        ax_a.axhline(level, color="#dc2626", lw=1, ls="--")
    ax_a.set_xlim(0, p[-1])
    ax_a.set_ylim(-1.15, 1.15)
    ax_a.set_xlabel("$p$")
    ax_a.set_ylabel(r"$a_p / 2\sqrt{p}$")
    ax_a.set_title(f"{names[pick]}: $N_p = p + 1 - a_p$ inside the Hasse bound", fontsize=9)

    edges = np.linspace(-1, 1, 25)
    hist = ax_h.stairs(np.zeros(len(edges) - 1), edges, fill=True, color="#93c5fd", label="all five curves")
    grid = np.linspace(-1, 1, 200)
    ax_h.plot(
        grid, 2 / np.pi * np.sqrt(1 - grid**2), color=ACCENT[1], lw=1.5, label=r"Sato–Tate $\frac{2}{\pi}\sqrt{1-x^2}$"
    )
    ax_h.set_xlim(-1, 1)
    ax_h.set_ylim(0, 1.0)
    ax_h.set_xlabel(r"$x = a_p / 2\sqrt{p}$")
    ax_h.legend(fontsize=7, loc="lower center")
    title_h = ax_h.set_title("", fontsize=9)

    s, curve = lf["s"].cpu().numpy(), lf["L"].cpu().numpy()
    for c, name in enumerate(names):
        ax_l.plot(s, curve[c], color=ACCENT[c], lw=1.4, label=f"{name}: order {int(lf['order'][c])}")
    ax_l.axhline(0, color="#9ca3af", lw=0.8)
    ax_l.axvline(1, color="#9ca3af", lw=0.8, ls="--")
    ax_l.set_ylim(-0.6, 1.6)
    ax_l.set_xlabel("$s$")
    ax_l.set_ylabel("$L(E, s)$")
    ax_l.legend(fontsize=7, loc="upper left")
    ax_l.set_title("$L(E, s)$ vanishes at $s = 1$ to order = rank", fontsize=9)

    def update(f: int):
        top = frames[f]
        k = int(np.searchsorted(p, top, side="right"))
        window = (p >= top**0.5) & (p <= top)
        for c, line in enumerate(lines):
            line.set_data(loglog[:k], log_prod[c, :k])
            if window.sum() > 3:
                est = _slope(loglog[window], log_prod[c, window])
                legend.get_texts()[c].set_text(f"{names[c]}, rank {ranks[c]}: slope {est:.2f}")
        title_g.set_text(f"growth like $(\\log X)^r$ — primes up to X = {int(top):,}")
        dots.set_offsets(np.c_[p[:k], x[pick, :k]])
        hist.set_data(np.histogram(x[:, :k].ravel(), bins=edges, density=True)[0])
        title_h.set_text(f"Sato–Tate distribution ({5 * k:,} values)")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(frames), len(frames)), path, fps=6)
