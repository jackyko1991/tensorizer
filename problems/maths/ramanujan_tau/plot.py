"""The Sato–Tate histogram of x_p = τ(p)/(2p^{11/2}) filling up against the semicircle, inside Deligne's bound."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

N_FRAMES = 40


def animate(d: dict, path: Path) -> Path:
    p = d["p"].cpu().numpy()
    x = d["x"].cpu().numpy()
    target = d["target"].cpu().numpy()
    count = np.arange(1, len(x) + 1)
    m2 = np.cumsum(x**2) / count  # running moments over the first n primes
    m4 = np.cumsum(x**4) / count

    fig, (ax_h, ax_s, ax_m) = plt.subplots(1, 3, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1.2, 1]})

    bins = np.linspace(-1, 1, 41)
    centres = 0.5 * (bins[1:] + bins[:-1])
    xs = np.linspace(-1, 1, 400)
    bars = ax_h.bar(
        centres, np.zeros_like(centres), width=bins[1] - bins[0], color="#93c5fd", edgecolor="#1e40af", lw=0.5
    )
    ax_h.plot(xs, 2 / np.pi * np.sqrt(1 - xs**2), color=ACCENT[1], lw=2, label=r"Sato–Tate $\frac{2}{\pi}\sqrt{1-x^2}$")
    ax_h.set_xlim(-1.05, 1.05)
    ax_h.set_ylim(0, 1.05)
    ax_h.set_xlabel(r"$x_p = \tau(p)\,/\,2p^{11/2}$")
    ax_h.set_ylabel("density")
    ax_h.legend(fontsize=8, loc="upper right")
    title_h = ax_h.set_title("", fontsize=10)

    sc = ax_s.scatter(p[:1], x[:1], s=2, color=ACCENT[0], lw=0)
    for y in (-1, 1):
        ax_s.axhline(y, color=ACCENT[1], lw=1.2, ls="--")
    ax_s.text(p[-1], 1.04, "Deligne bound", ha="right", va="bottom", fontsize=8, color=ACCENT[1])
    ax_s.set_xscale("log")
    ax_s.set_xlim(2, p[-1] * 1.05)
    ax_s.set_ylim(-1.15, 1.2)
    ax_s.set_xlabel("prime $p$")
    ax_s.set_ylabel("$x_p$")
    ax_s.set_title(r"$|\tau(p)| \leq 2p^{11/2}$ (Deligne 1974)", fontsize=10)

    (l2,) = ax_m.semilogx(count[:1], m2[:1], color=ACCENT[0], lw=1.5, label=r"$\langle x^2\rangle$")
    (l4,) = ax_m.semilogx(count[:1], m4[:1], color=ACCENT[2], lw=1.5, label=r"$\langle x^4\rangle$")
    ax_m.axhline(target[0], color=ACCENT[0], lw=1, ls=":", label="1/4")
    ax_m.axhline(target[1], color=ACCENT[2], lw=1, ls=":", label="1/8")
    ax_m.set_xlim(1, len(x))
    ax_m.set_ylim(0, 0.45)
    ax_m.set_xlabel("number of primes")
    ax_m.legend(fontsize=8, loc="upper right", ncol=2)
    ax_m.set_title("moments → semicircle values", fontsize=10)

    fig.text(
        0.5,
        0.01,
        "Ramanujan noticed that… |τ(p)| ≤ 2p^(11/2) (1916, conjectured). Deligne proved it in 1974.",
        ha="center",
        fontsize=9,
        color="#6b7280",
        style="italic",
    )

    def update(f: int):
        s = (f + 1) / N_FRAMES
        n = max(10, int(round(len(x) ** s)))
        h, _ = np.histogram(x[:n], bins=bins, density=True)
        for bar, height in zip(bars, h):
            bar.set_height(height)
        title_h.set_text(f"{n:,} primes up to {p[n - 1]:,}")
        sc.set_offsets(np.column_stack([p[:n], x[:n]]))
        l2.set_data(count[:n], m2[:n])
        l4.set_data(count[:n], m4[:n])

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_gif(fig, update, frame_indices(N_FRAMES, N_FRAMES), path, fps=6)
