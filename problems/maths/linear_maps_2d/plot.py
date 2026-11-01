"""Animate the plane being transformed (left) with det and the singular values along the path (right)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

LIM = 3.2


def animate(
    m: torch.Tensor,
    which: torch.Tensor,
    t: torch.Tensor,
    y_lines: torch.Tensor,
    y_circle: torch.Tensor,
    y_square: torch.Tensor,
    u: torch.Tensor,
    sigma: torch.Tensor,
    d: torch.Tensor,
    eig: list[torch.Tensor],
    captions: list[str],
    path: Path,
) -> Path:
    """m (T, 2, 2); which, t (T,); y_lines (T, L, S, 2); y_circle (T, K, 2); y_square (T, 5, 2); u (T, 2, 2);
    sigma (T, 2); d (T,); eig[s] = eigenvectors of matrix s in the columns (2, k)."""
    m, which, t, y_lines, y_circle, y_square, u, sigma, d = (
        x.cpu().numpy() for x in (m, which, t, y_lines, y_circle, y_square, u, sigma, d)
    )
    eig = [v.cpu().numpy() for v in eig]
    frames_all = np.arange(len(t))

    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(8, 4.1), gridspec_kw={"width_ratios": [1.15, 1]})
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axhline(0, color="#d1d5db", lw=0.8)
    ax.axvline(0, color="#d1d5db", lw=0.8)
    grid = [ax.plot([], [], color="#93c5fd", lw=0.7)[0] for _ in range(y_lines.shape[1])]
    eig_lines = [ax.plot([], [], color="#f59e0b", lw=1.2, ls="--")[0] for _ in range(2)]
    eig_dots = [ax.plot([], [], "o", color="#f59e0b", ms=6, mec="#b45309")[0] for _ in range(2)]
    (square,) = ax.fill([], [], color="#fde68a", lw=1.5, ec="#b45309", zorder=1)
    (circle,) = ax.plot([], [], color="#2563eb", lw=2)
    svd_axes = [ax.plot([], [], color="#9333ea", lw=1.8, ls=ls)[0] for ls in ("-", "--")]
    arrows: list = []
    caption = ax.set_title("two lines\nof caption", fontsize=10)  # placeholder so tight_layout leaves room

    ax_r.plot(frames_all, d, color="#374151", lw=1.6, label="det M")
    ax_r.plot(frames_all, sigma[:, 0], color="#9333ea", lw=1.2, label="σ₁ (long axis)")
    ax_r.plot(frames_all, sigma[:, 1], color="#9333ea", lw=1.2, ls="--", label="σ₂ (short axis)")
    ax_r.axhline(0, color="#9ca3af", lw=0.8)
    for s in range(len(captions)):  # shade every other matrix
        if s % 2:
            span = frames_all[which == s]
            ax_r.axvspan(span[0], span[-1], color="#f3f4f6", lw=0)
    marker = ax_r.axvline(0, color="#dc2626", lw=1.2)
    ax_r.set_xlim(0, len(t) - 1)
    ax_r.set_xlabel("frame (each block: I → A)")
    ax_r.set_title("|det| = area factor = σ₁σ₂", fontsize=10)
    ax_r.legend(fontsize=8, loc="lower left")

    def arrow(start, vec, color):
        return ax.annotate(
            "", xy=start + vec, xytext=start, arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8}
        )

    def update(k: int):
        s = which[k]
        for line, pts in zip(grid, y_lines[k]):
            line.set_data(pts[:, 0], pts[:, 1])
        circle.set_data(y_circle[k, :, 0], y_circle[k, :, 1])
        square.set_xy(y_square[k])
        square.set_facecolor("#fecaca" if d[k] < 0 else "#fde68a")
        vecs = eig[s]
        for j in range(2):
            if j < vecs.shape[1]:
                v = vecs[:, j]
                eig_lines[j].set_data([-5 * v[0], 5 * v[0]], [-5 * v[1], 5 * v[1]])
                mv = m[k] @ v  # M(t) v stays on the line through v
                eig_dots[j].set_data([mv[0]], [mv[1]])
            else:
                eig_lines[j].set_data([], [])
                eig_dots[j].set_data([], [])
        while arrows:
            arrows.pop().remove()
        origin = np.zeros(2)
        arrows.append(arrow(origin, m[k][:, 0], "#dc2626"))  # image of e₁ (first column)
        arrows.append(arrow(origin, m[k][:, 1], "#16a34a"))  # image of e₂ (second column)
        for j in range(2):  # the ellipse's axes run from −σ_j u_j to +σ_j u_j
            end = sigma[k, j] * u[k][:, j]
            svd_axes[j].set_data([-end[0], end[0]], [-end[1], end[1]])
        marker.set_xdata([k, k])
        caption.set_text(f"{captions[s]}\nt = {t[k]:.2f},  det = {d[k]:+.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 120), path, fps=10)
