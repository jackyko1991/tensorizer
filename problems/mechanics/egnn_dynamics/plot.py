"""Animate the rollout (true vs EGNN vs MLP, seen from above) with the rollout error and the training curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

GREY, BLUE, ORANGE, RED = "#6b7280", "#2563eb", "#ea580c", "#dc2626"


def animate(true: torch.Tensor, pred: dict, charges: torch.Tensor, results: dict, dt: float, path: Path) -> Path:
    tr = true.cpu().numpy()  # (T, N, 3)
    eg, ml = pred["egnn"].cpu().numpy(), pred["mlp"].cpu().numpy()
    T = len(tr)
    t = np.arange(T) * dt
    err_eg = np.linalg.norm(eg - tr, axis=-1).mean(-1)  # (T,)
    err_ml = np.linalg.norm(ml - tr, axis=-1).mean(-1)
    q = charges.cpu().numpy()

    fig = plt.figure(figsize=(8.4, 4.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax_e = fig.add_subplot(grid[0, 1])
    ax_l = fig.add_subplot(grid[1, 1])

    # --- rollout, projected on the x–y plane ---
    lim = np.abs(tr[..., :2]).max() * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("rollout (top view): true vs predicted", fontsize=10)
    face = np.where(q > 0, RED, BLUE)
    trails = [ax.plot([], [], color=GREY, lw=1, alpha=0.6)[0] for _ in range(tr.shape[1])]
    dots_true = ax.scatter(
        tr[0, :, 0], tr[0, :, 1], s=70, c=face, edgecolors="#111827", zorder=4, label="true (± charge)"
    )
    dots_eg = ax.scatter(eg[0, :, 0], eg[0, :, 1], s=150, facecolors="none", edgecolors=BLUE, lw=1.8, zorder=5,
                         label="EGNN")  # fmt: skip
    dots_ml = ax.scatter(ml[0, :, 0], ml[0, :, 1], s=60, marker="x", color=ORANGE, lw=1.8, zorder=5, label="MLP")
    ax.legend(loc="lower left", fontsize=7)
    clock = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top", fontsize=9)

    # --- rollout error ---
    ax_e.set_xlim(0, t[-1])
    floor = max(1e-4, min(err_eg[1:].min(), err_ml[1:].min()) * 0.5) if T > 1 else 1e-4
    ax_e.set_yscale("log")
    ax_e.minorticks_off()
    ax_e.set_ylim(floor, max(err_eg.max(), err_ml.max(), floor) * 2)
    ax_e.set_title("rollout: mean position error", fontsize=10)
    ax_e.tick_params(labelsize=8)
    (line_eg,) = ax_e.plot([], [], color=BLUE, lw=1.8, label="EGNN")
    (line_ml,) = ax_e.plot([], [], color=ORANGE, lw=1.8, label="MLP")
    ax_e.legend(loc="upper left", fontsize=7)

    # --- training curves (static) ---
    for name, color in (("egnn", BLUE), ("mlp", ORANGE)):
        h = np.array(results[name]["train"])
        ax_l.plot(np.arange(1, len(h) + 1), h, color=color, lw=1.2, label=f"{name.upper()} train")
        ax_l.axhline(results[name]["test_final"], color=color, ls="--", lw=1, label=f"{name.upper()} test")
    ax_l.set_yscale("log")
    ax_l.minorticks_off()
    ax_l.set_xlabel("iteration", fontsize=9)
    ax_l.set_title("one-step MSE", fontsize=10)
    ax_l.tick_params(labelsize=8)
    ax_l.legend(loc="upper right", fontsize=6, ncol=2)

    def update(s: int):
        lo = max(0, s - 15)
        for n, trail in enumerate(trails):
            trail.set_data(tr[lo : s + 1, n, 0], tr[lo : s + 1, n, 1])
        dots_true.set_offsets(tr[s, :, :2])
        dots_eg.set_offsets(eg[s, :, :2])
        dots_ml.set_offsets(ml[s, :, :2])
        line_eg.set_data(t[: s + 1], err_eg[: s + 1])
        line_ml.set_data(t[: s + 1], err_ml[: s + 1])
        clock.set_text(f"t = {t[s]:.1f}  (step {s})")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(T, 80), path, fps=12)
