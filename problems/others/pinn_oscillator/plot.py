"""Animate training: the network's guess x̂(t) closing in on the true motion, the loss terms, and μ (or the error)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

BLUE, ORANGE, RED, GREY = "#2563eb", "#ea580c", "#dc2626", "#9ca3af"


def animate(
    t_plot: torch.Tensor,
    x_exact: torch.Tensor,
    t_d: torch.Tensor,
    x_d: torch.Tensor,
    t_c: torch.Tensor,
    pinn: dict,
    plain: dict | None,
    lam_phys: float,
    mu_true: float,
    inverse: bool,
    path: Path,
) -> Path:
    t, ex = t_plot.cpu().numpy(), x_exact.cpu().numpy()
    snaps = pinn["snapshots"].numpy()  # (S, n_plot)
    snaps_plain = plain["snapshots"].numpy() if plain is not None else None
    iters = np.array(pinn["snap_iters"])  # iteration of each snapshot
    n_iter = len(pinn["mu"])
    stride = max(1, n_iter // 400)  # ~400 points per history curve are plenty for drawing
    it_hist = np.arange(0, n_iter, stride)
    loss = pinn["losses"][::stride].numpy()  # (n_hist, 3): data, physics, initial
    t_end = t_d.max().item()

    fig = plt.figure(figsize=(8.2, 4.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax_l = fig.add_subplot(grid[0, 1])
    ax_m = fig.add_subplot(grid[1, 1])

    # --- the motion: exact (grey), data (dots), collocation points (ticks), PINN (bold), no physics (dashed) ---
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.3, 1.4)
    ax.set_xlabel("time t (s)")
    ax.set_ylabel("displacement x")
    ax.set_title("damped oscillator: network vs exact", fontsize=10)
    ax.axvspan(0, t_end, color="#dbeafe", alpha=0.6, lw=0)
    ax.text(t_end / 2, 1.3, "data", ha="center", va="top", fontsize=8, color="#1e40af")
    ax.text((t_end + t[-1]) / 2, 1.3, "extrapolation", ha="center", va="top", fontsize=8, color="#6b7280")
    ax.plot(t, ex, color=GREY, lw=3, alpha=0.7, label="exact")
    ax.plot(t_c.cpu().numpy(), np.full(len(t_c), -1.22), "|", color="#16a34a", ms=7, label="collocation $t_c$")
    ax.plot(t_d.cpu().numpy(), x_d.cpu().numpy(), "o", color=RED, ms=4.5, zorder=5, label="noisy data")
    (line_pinn,) = ax.plot([], [], color=BLUE, lw=2.2, label="PINN")
    line_plain = None
    if snaps_plain is not None:
        (line_plain,) = ax.plot([], [], "--", color=ORANGE, lw=1.6, label="no physics")
    ax.legend(loc="lower right", fontsize=7, ncol=2, bbox_to_anchor=(1.0, 0.04))
    counter = ax.text(0.03, 0.1, "", transform=ax.transAxes, fontsize=9)

    # --- loss terms vs iteration (log scale) ---
    ax_l.set_yscale("log")
    ax_l.set_xlim(0, n_iter)
    scale = np.array([1.0, lam_phys, 1.0])  # draw the physics term as it enters the loss, λ·mean(r²)
    shown = np.maximum(loss * scale, 1e-12)
    ax_l.set_ylim(shown.min() * 0.5, shown.max() * 2)
    ax_l.minorticks_off()
    ax_l.set_title("loss terms", fontsize=10)
    ax_l.tick_params(labelsize=8)
    curves = [
        ax_l.plot([], [], color=RED, lw=1.2, label="data")[0],
        ax_l.plot([], [], color="#16a34a", lw=1.2, label="λ·physics")[0],
        ax_l.plot([], [], color="#9333ea", lw=1.2, label="initial")[0],
    ]
    ax_l.legend(loc="upper right", fontsize=7)

    # --- inverse: inferred μ vs iteration;  forward: relative error vs iteration ---
    ax_m.set_xlim(0, n_iter)
    ax_m.set_xlabel("iteration", fontsize=9)
    ax_m.tick_params(labelsize=8)
    if inverse:
        mu = pinn["mu"][::stride].numpy()
        ax_m.set_title("inferred damping μ", fontsize=10)
        ax_m.axhline(mu_true, color=GREY, ls="--", lw=1)
        ax_m.text(n_iter * 0.97, mu_true, f"true {mu_true:g}", ha="right", va="bottom", fontsize=8, color="#6b7280")
        ax_m.set_ylim(0, max(mu.max(), mu_true) * 1.15)
        (mu_line,) = ax_m.plot([], [], color=BLUE, lw=1.6)
    else:
        err = np.linalg.norm(snaps - ex, axis=1) / np.linalg.norm(ex)
        err_plain = np.linalg.norm(snaps_plain - ex, axis=1) / np.linalg.norm(ex) if snaps_plain is not None else None
        ax_m.set_title("relative L2 error vs exact", fontsize=10)
        ax_m.set_yscale("log")
        ax_m.minorticks_off()
        lo = min(err.min(), err_plain.min() if err_plain is not None else 1.0)
        ax_m.set_ylim(lo * 0.5, max(err.max(), err_plain.max() if err_plain is not None else 0) * 2)
        (err_line,) = ax_m.plot([], [], color=BLUE, lw=1.6)
        err_plain_line = ax_m.plot([], [], "--", color=ORANGE, lw=1.4)[0] if err_plain is not None else None

    def update(s: int):
        line_pinn.set_data(t, snaps[s])
        if line_plain is not None:
            line_plain.set_data(t, snaps_plain[s])
        upto = it_hist <= iters[s]
        for j, curve in enumerate(curves):
            curve.set_data(it_hist[upto], shown[upto, j])
        if inverse:
            mu_line.set_data(it_hist[upto], mu[upto])
            counter.set_text(f"iteration {iters[s] + 1}   μ = {mu[upto][-1]:.2f}")
        else:
            err_line.set_data(iters[: s + 1], err[: s + 1])
            if err_plain_line is not None:
                err_plain_line.set_data(iters[: s + 1], err_plain[: s + 1])
            counter.set_text(f"iteration {iters[s] + 1}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(snaps), 50), path)
