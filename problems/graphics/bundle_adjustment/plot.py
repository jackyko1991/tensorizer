"""Animate Levenberg–Marquardt: the scene converging to the truth, one camera's reprojection errors, the block
structure of JᵀJ, the RMS error, the residual histogram and the distance to the truth."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import save_gif

from .tensors import IMAGE

BLUE, RED, GREY, DARK, ORANGE = "#2563eb", "#dc2626", "#9ca3af", "#374151", "#f59e0b"


def animate(frames, truth, observed, costs, errs, H, n_cam, sigma, floor, show_cam, path: Path) -> Path:
    m = frames[0]["r"].numel()
    rms = np.sqrt(2 * np.array(costs) / m)
    its = np.arange(len(frames))
    e_pts, e_cam = np.array([e[0] for e in errs]), np.array([e[1] for e in errs])
    obs = observed.numpy()

    fig = plt.figure(figsize=(9.0, 5.4))
    ax3 = fig.add_subplot(2, 3, 1, projection="3d")
    ax_i, ax_h = fig.add_subplot(2, 3, 2), fig.add_subplot(2, 3, 3)
    ax_r, ax_hist, ax_e = fig.add_subplot(2, 3, 4), fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)

    # --- 3D scene: truth in grey, estimate in colour (x right, z forward, y down → plot with -y up) ---
    def xyz(p):
        p = p.numpy()
        return p[:, 0], p[:, 2], -p[:, 1]

    ax3.scatter(*xyz(truth["X"]), s=6, color=GREY, depthshade=False, label="true")
    ax3.scatter(*xyz(truth["centre"]), s=40, facecolors="none", edgecolors=DARK, marker="^", depthshade=False)
    pts = ax3.scatter(*xyz(frames[0]["X"]), s=6, color=BLUE, depthshade=False, label="estimate")
    cams = ax3.scatter(*xyz(frames[0]["centre"]), s=25, color=RED, marker="^", depthshade=False)
    axis_lines = [ax3.plot([], [], [], color=RED, lw=1)[0] for _ in range(len(truth["centre"]))]
    lim = 1.1 * truth["centre"].abs().max().item()
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, 1.5)
    ax3.set_zlim(-2, 2)
    ax3.view_init(elev=55, azim=-80)
    ax3.set_title("cameras (▲) and points", fontsize=8.5)
    ax3.tick_params(labelsize=6)
    ax3.legend(fontsize=6, loc="upper left")

    # --- one camera's image: measured points and the current reprojections ---
    ax_i.scatter(obs[:, 0], obs[:, 1], s=7, color=DARK, zorder=3, label="measured")
    proj = ax_i.scatter(*frames[0]["proj"].numpy().T, s=10, facecolors="none", edgecolors=BLUE, lw=0.8,
                        label="reprojected $\\pi(R X + t)$")  # fmt: skip
    res_lines = [ax_i.plot([], [], color=RED, lw=0.8)[0] for _ in range(len(obs))]
    ax_i.set_xlim(0, IMAGE)
    ax_i.set_ylim(IMAGE, 0)
    ax_i.set_aspect("equal")
    ax_i.set_title(f"image of camera {show_cam}: residuals (red)", fontsize=8.5)
    ax_i.tick_params(labelsize=6)
    ax_i.legend(fontsize=6, loc="lower right")

    # --- the sparsity of JᵀJ (static) ---
    Hn = H.abs().numpy()
    ax_h.imshow(np.log10(Hn / Hn.max() + 1e-12), cmap="Blues", vmin=-8, vmax=0, interpolation="nearest")
    ax_h.axhline(n_cam - 0.5, color=RED, lw=0.8)
    ax_h.axvline(n_cam - 0.5, color=RED, lw=0.8)
    ax_h.text(n_cam / 2, n_cam / 2, "U", ha="center", va="center", fontsize=10, color=RED)
    ax_h.text(n_cam + (Hn.shape[0] - n_cam) / 2, n_cam / 2, "W", ha="center", va="center", fontsize=10, color=RED)
    ax_h.text(n_cam + (Hn.shape[0] - n_cam) / 2, n_cam + (Hn.shape[0] - n_cam) / 2, "V: 3×3 blocks", ha="center",
              va="center", fontsize=7.5, color=RED)  # fmt: skip
    ax_h.set_title("$|J^TJ|$: cameras | points", fontsize=8.5)
    ax_h.set_xticks([])
    ax_h.set_yticks([])
    ax_h.grid(False)

    # --- RMS reprojection error ---
    ax_r.semilogy(its, rms, color=GREY, lw=0.8)
    (rms_line,) = ax_r.semilogy([], [], "o-", color=BLUE, ms=3, lw=1.3)
    ax_r.axhline(floor, color=ORANGE, ls="--", lw=1)
    ax_r.text(its[-1], floor * 1.15, "noise floor", ha="right", va="bottom", fontsize=7, color="#b45309")
    ax_r.set_xlim(0, its[-1])
    ax_r.set_title("RMS reprojection error (px)", fontsize=8.5)
    ax_r.set_xlabel("LM iteration", fontsize=7.5)
    ax_r.tick_params(labelsize=7)
    ax_r.minorticks_off()

    # --- residual histogram against the noise model N(0, σ²) ---
    edge = 5 * sigma
    bins = np.linspace(-edge, edge, 31)
    xs = np.linspace(-edge, edge, 200)
    ax_hist.plot(xs, np.exp(-0.5 * (xs / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)), color=ORANGE, lw=1.3,
                 label=f"noise $N(0, {sigma:g}^2)$")  # fmt: skip
    ax_hist.set_xlim(-edge, edge)
    ax_hist.set_title("residuals $r_{cpa}$ (px), clipped at ±5σ", fontsize=8.5)
    ax_hist.tick_params(labelsize=7)
    ax_hist.legend(fontsize=6.5, loc="upper right")
    hist_bars = {"bars": None}

    # --- distance to the truth ---
    ax_e.semilogy(its, e_pts, color=GREY, lw=0.8)
    ax_e.semilogy(its, e_cam, color=GREY, lw=0.8)
    (ep_line,) = ax_e.semilogy([], [], "o-", color=BLUE, ms=3, lw=1.3, label="points")
    (ec_line,) = ax_e.semilogy([], [], "^-", color=RED, ms=3, lw=1.3, label="camera centres")
    ax_e.set_xlim(0, its[-1])
    ax_e.set_title("RMS distance to the truth", fontsize=8.5)
    ax_e.set_xlabel("LM iteration", fontsize=7.5)
    ax_e.tick_params(labelsize=7)
    ax_e.legend(fontsize=6.5)
    ax_e.minorticks_off()
    counter = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9)

    def update(k: int):
        fr = frames[k]
        pts._offsets3d = xyz(fr["X"])
        cams._offsets3d = xyz(fr["centre"])
        for line, ctr, R in zip(axis_lines, fr["centre"], fr["R"]):
            seg = torch.stack([ctr, ctr + 1.2 * R[2]])  # optical axis = third row of R
            x, y, z = xyz(seg)
            line.set_data_3d(x, y, z)
        p = fr["proj"].numpy()
        proj.set_offsets(p)
        for line, a, b in zip(res_lines, obs, p):
            line.set_data([a[0], b[0]], [a[1], b[1]])
        if hist_bars["bars"] is not None:
            hist_bars["bars"].remove()
        r = np.clip(fr["r"].reshape(-1).numpy(), -edge * 0.999, edge * 0.999)
        hist_bars["bars"] = ax_hist.hist(r, bins=bins, density=True, color=BLUE, alpha=0.55)[2]
        top = max(1.25 / (sigma * np.sqrt(2 * np.pi)), max(b.get_height() for b in hist_bars["bars"]) * 1.05)
        ax_hist.set_ylim(0, top)
        rms_line.set_data(its[: k + 1], rms[: k + 1])
        ep_line.set_data(its[: k + 1], e_pts[: k + 1])
        ec_line.set_data(its[: k + 1], e_cam[: k + 1])
        counter.set_text(f"iteration {k}   ·   RMS reprojection error {rms[k]:.3f} px   ·   noise σ = {sigma:g} px")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_gif(fig, update, list(range(len(frames))), path, fps=2)
