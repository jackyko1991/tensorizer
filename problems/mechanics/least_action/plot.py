"""Animate the trial paths relaxing onto the true path, with −∂S/∂q_n arrows and the action per iteration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(t: torch.Tensor, hist: dict, exact: torch.Tensor, S_exact: torch.Tensor, path: Path) -> Path:
    t = t.cpu().numpy()  # (N+1,)
    paths = hist["paths"].cpu().numpy()  # (K+1, B, N+1)
    S = hist["S"].cpu().numpy()  # (K+1, B)
    grad = hist["grad"].cpu().numpy()  # (K+1, B, N−1)
    exact = exact.cpu().numpy()
    S_exact = float(S_exact)
    n_iter, n_paths = S.shape[0] - 1, S.shape[1]
    iters = np.arange(n_iter + 1)

    fig, (ax, ax_s) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1.3, 1]})

    # --- left: the paths q(t) ---
    lo, hi = min(paths[0].min(), exact.min()), max(paths[0].max(), exact.max())
    pad = 0.08 * (hi - lo)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("q (m)")
    ax.plot(t, exact, color="#111827", lw=3, alpha=0.35, label="true path")
    lines = [ax.plot([], [], "-o", ms=2, lw=1.3, color=ACCENT[b % len(ACCENT)])[0] for b in range(n_paths)]
    ax.plot(t[[0, -1]], exact[[0, -1]], "o", ms=8, color="#374151", zorder=5)
    # arrows: the downhill direction −∂S/∂q_n on the last (most wiggly) path, scaled by the starting gradient
    arrow_scale = 0.12 * (hi - lo) / (np.abs(grad[0, -1]).max() + 1e-30)
    arrows = ax.quiver(
        t[1:-1],
        paths[0, -1, 1:-1],
        np.zeros(len(t) - 2),
        -grad[0, -1] * arrow_scale,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=ACCENT[(n_paths - 1) % len(ACCENT)],
        width=0.004,
        alpha=0.7,
    )
    ax.legend(loc="lower center", fontsize=8)
    title = ax.set_title("iteration 0", fontsize=9)

    # --- right: the action of every path vs iteration ---
    excess = max(S[0].max() - S_exact, 1e-9)
    ax_s.set_xscale("log")
    ax_s.set_xlim(1, n_iter + 1)
    ax_s.set_ylim(S_exact - 0.08 * excess, S_exact + 1.05 * excess)
    ax_s.axhline(S_exact, color="#111827", lw=1.2, ls="--", label="S of the true path")
    s_lines = [ax_s.plot([], [], lw=1.3, color=ACCENT[b % len(ACCENT)])[0] for b in range(n_paths)]
    ax_s.set_xlabel("iteration + 1")
    ax_s.set_ylabel("action S (J·s)")
    ax_s.legend(loc="upper right", fontsize=8)

    def update(k: int):
        for b in range(n_paths):
            lines[b].set_data(t, paths[k, b])
            s_lines[b].set_data(iters[: k + 1] + 1, S[: k + 1, b])
        arrows.set_offsets(np.column_stack([t[1:-1], paths[k, -1, 1:-1]]))
        arrows.set_UVC(np.zeros(len(t) - 2), -grad[k, -1] * arrow_scale)
        title.set_text(f"iteration {k}:  max |∂S/∂q_n| = {np.abs(grad[k]).max():.1e}")

    # most of the change happens early: space the frames evenly in log(iteration)
    picks = frame_indices(n_iter + 1, 80)
    frames = [round((n_iter + 1) ** (p / max(1, n_iter))) - 1 for p in picks]
    fig.tight_layout()
    return save_gif(fig, update, frames, path)
