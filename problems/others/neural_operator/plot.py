"""Animate training: FNO predictions on unseen initial conditions, the error curves, and zero-shot super-resolution."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif

BLUE, RED, ORANGE, GREY, DARK = "#2563eb", "#dc2626", "#ea580c", "#9ca3af", "#374151"


def animate(data: dict, hist: dict, baseline: float, fine_err: float, path: Path) -> Path:
    x = data["x"].cpu().numpy()  # (N,)
    xf = data["x_fine"].cpu().numpy()  # (M,)
    show = [0, 1, 2]  # the first three test cases
    u0 = data["u0_test"][show].cpu().numpy()  # (3, N)
    uT = data["uT_test"][show].cpu().numpy()
    k = data["fine_idx"]  # the test case with the steepest front
    uT_fine = data["uT_test_fine"][k].cpu().numpy()  # (M,) at full resolution
    pred_coarse = hist["pred_test"][:, k].numpy()  # (E+1, N) the same case on the training grid
    pred = hist["pred_test"][:, show].numpy()  # (E+1, 3, N)
    pred_fine = hist["pred_fine"][:, 0].numpy()  # (E+1, M)
    train_err, test_err = hist["train_err"].numpy(), hist["test_err"].numpy()
    epochs = np.arange(len(train_err))
    lim = 1.15 * max(np.abs(u0).max(), np.abs(uT).max())

    fig = plt.figure(figsize=(8.0, 4.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
    tops = [fig.add_subplot(grid[0, j]) for j in range(3)]
    ax_e = fig.add_subplot(grid[1, 0])
    ax_s = fig.add_subplot(grid[1, 1:])

    # --- three unseen test initial conditions: u0 (dashed), true u(T) (grey), FNO (blue) ---
    lines = []
    for j, ax in enumerate(tops):
        ax.plot(x, u0[j], "--", color=GREY, lw=1.2, label="$u_0$")
        ax.plot(x, uT[j], color=DARK, lw=3, alpha=0.35, label="true $u(T)$")
        lines.append(ax.plot([], [], color=BLUE, lw=1.6, label="FNO")[0])
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
        ax.tick_params(labelsize=8)
        ax.set_title(f"unseen test case {j + 1}", fontsize=9)
    tops[0].legend(loc="lower right", fontsize=6.5)

    # --- relative L2 error vs epoch ---
    ax_e.set_yscale("log")
    ax_e.minorticks_off()
    ax_e.set_xlim(0, epochs[-1])
    lo = min(train_err.min(), test_err.min())
    ax_e.set_ylim(lo * 0.6, max(train_err.max(), test_err.max(), baseline) * 1.6)
    ax_e.axhline(baseline, color=GREY, ls="--", lw=1)
    ax_e.text(epochs[-1] * 0.97, baseline * 1.1, "$u(T) = u_0$", ha="right", va="bottom", fontsize=7, color="#6b7280")
    (train_line,) = ax_e.plot([], [], color=BLUE, lw=1.4, label="train")
    (test_line,) = ax_e.plot([], [], color=RED, lw=1.4, label="test")
    ax_e.set_xlabel("epoch", fontsize=8)
    ax_e.set_title("relative $L^2$ error", fontsize=9)
    ax_e.tick_params(labelsize=7)
    ax_e.legend(loc="center right", fontsize=7)

    # --- zero-shot super-resolution: same weights on 256 points ---
    ax_s.plot(xf, uT_fine, color=DARK, lw=3, alpha=0.35, label=f"solver, {len(xf)} pts")
    (fine_line,) = ax_s.plot([], [], color=BLUE, lw=1.4, label=f"FNO on {len(xf)} pts (zero-shot)")
    (coarse_pts,) = ax_s.plot([], [], "o", color=ORANGE, ms=3, label=f"FNO on {len(x)} pts (training grid)")
    ax_s.set_xlim(0, 2 * np.pi)
    lim_s = 1.15 * np.abs(uT_fine).max()
    ax_s.set_ylim(-lim_s, lim_s)
    ax_s.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
    ax_s.tick_params(labelsize=7)
    ax_s.set_title(f"super-resolution: steepest test case, trained on {len(x)} points, run on {len(xf)}", fontsize=9)
    ax_s.legend(loc="upper left", fontsize=6.5)
    case_err = np.linalg.norm(pred_fine[-1] - uT_fine) / np.linalg.norm(uT_fine)
    counter = fig.text(0.5, 0.005, "", ha="center", va="bottom", fontsize=9)

    def update(e: int):
        for j, line in enumerate(lines):
            line.set_data(x, pred[e, j])
        train_line.set_data(epochs[: e + 1], train_err[: e + 1])
        test_line.set_data(epochs[: e + 1], test_err[: e + 1])
        fine_line.set_data(xf, pred_fine[e])
        coarse_pts.set_data(x, pred_coarse[e])
        text = f"epoch {e}   ·   test error {test_err[e]:.1%}"
        if e == len(train_err) - 1:  # the super-resolution errors refer to the final model
            n_test = len(data["u0_test"])
            text += f"   ·   on {len(xf)} pts: all {n_test} test cases {fine_err:.1%}, steepest {case_err:.1%}"
        counter.set_text(text)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_gif(fig, update, frame_indices(len(train_err), 30), path, fps=6)
