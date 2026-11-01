"""Animate the object sliding towards the lens: principal rays, the image and its type (real / virtual)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(s: torch.Tensor, h: float, f: float, res: dict, path: Path) -> Path:
    s = s.numpy()  # (K,)
    s_img, m, h_img = res["s_img"].numpy(), res["m"].numpy(), res["h_img"].numpy()  # (K,) each
    y, slope_out = res["y"].numpy(), res["slope_out"].numpy()  # (K, 3) each
    x_min, x_max = -1.1 * s.max(), max(3.5 * f, 0.5 * s.max())
    y_max = 3.5 * h

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("height (cm)")
    ax.axhline(0, color="#6b7280", lw=0.8)
    lens_kw = {"arrowstyle": "<|-|>", "color": "#2563eb", "lw": 2, "mutation_scale": 12}
    ax.annotate("", (0, 0.95 * y_max), (0, -0.95 * y_max), arrowprops=lens_kw)
    for x, name in [(-2 * f, "2F"), (-f, "F"), (f, "F"), (2 * f, "2F")]:
        ax.plot([x], [0], "k|", ms=8)
        ax.text(x, -0.1 * h, name, ha="center", va="top", fontsize=8)

    colors = ["#dc2626", "#16a34a", "#9333ea"]  # parallel, centre, focal ray
    names = ["parallel → F", "through centre", "through F → parallel"]
    before = [ax.plot([], [], lw=1.3, color=c)[0] for c in colors]
    after = [ax.plot([], [], lw=1.3, color=c, label=n)[0] for c, n in zip(colors, names)]
    behind = [ax.plot([], [], ":", lw=1.0, color=c)[0] for c in colors]  # back-extensions to a virtual image
    arrow_kw = {"arrowstyle": "-|>", "lw": 2.5, "mutation_scale": 14}
    obj = ax.annotate("", (0, 0), (0, 0), arrowprops={**arrow_kw, "color": "#111827"})
    img = ax.annotate("", (0, 0), (0, 0), arrowprops={**arrow_kw, "color": "#ea580c"})
    ax.legend(loc="lower right", fontsize=7)
    title = ax.set_title(" ")

    def update(k: int):
        obj.xy, obj.xyann = (-s[k], h), (-s[k], 0)  # object arrow from the axis up to its tip
        img.xy, img.xyann = (s_img[k], h_img[k]), (s_img[k], 0)
        real = s_img[k] > 0
        img.arrow_patch.set_linestyle("-" if real else "--")
        for r in range(3):
            before[r].set_data([-s[k], 0], [h, y[k, r]])  # object tip → lens
            after[r].set_data([0, x_max], [y[k, r], y[k, r] + slope_out[k, r] * x_max])  # lens → right edge
            if real:
                behind[r].set_data([], [])
            else:  # continue the outgoing ray backwards to the virtual image at x = s' < 0
                behind[r].set_data([0, s_img[k]], [y[k, r], y[k, r] + slope_out[k, r] * s_img[k]])
        kind = "real, inverted" if real else "virtual, upright"
        title.set_text(f"s = {s[k]:.1f} cm   s' = {s_img[k]:.1f} cm   m = {m[k]:.2f}   ({kind})")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(s), 90), path, fps=10)
