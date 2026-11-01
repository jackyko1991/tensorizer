"""Animate the experiment: molecules crossing the focus, the photon trace, the lifetime histogram and G(τ)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from problems._common.plotting import frame_indices, save_gif

from . import equation

SLICE = 0.35  # molecules with |y| < SLICE µm are drawn in the x–z slice


def animate(r: dict, n_build: int, n_hold: int, frame_dt: float, path: Path) -> Path:
    sp = r["sp"]
    n_s = len(sp["keys"])
    colors = sp["colors"]
    w0, z0 = r["w0"], r["z0"]
    pos = r["positions"].cpu().numpy()  # (R, n, 3) one row per frame
    who = r["S"].argmax(1).cpu().numpy()  # species of each molecule
    W = equation.detection(r["positions"], r["A"]).cpu().numpy()  # (R, n) brightness of each molecule
    fine = r["fine_counts"].cpu().numpy()  # (R, s+1) photons per frame interval
    lag = r["lag"].cpu().numpy() * 1e3  # ms
    G_snap = [g.cpu().numpy() for g in r["G_snap"]]
    h_snap = [h.cpu().numpy() for h in r["h_snap"]]
    t_ns = r["t"].cpu().numpy()
    n_frames = n_build + n_hold
    n_snap = len(G_snap)

    # final fits as curves
    kappa = r["kappa"]
    lag_s = r["lag"]
    fit_flcs = (equation.diffusion_model(lag_s, r["tauD_flcs"], kappa) / r["N_flcs"]).cpu().numpy()  # (L, s)
    fit_all = (equation.diffusion_model(lag_s, r["tauD_all"], kappa) @ r["a_all"]).cpu().numpy()  # (L,)
    comps = (r["M"][:, :n_s] * r["photons"]).cpu().numpy()  # (C, s) fitted decay of each species
    flim_fit = comps.sum(1) + r["bg_photons"].item() / len(t_ns)
    D_flcs = (w0**2 / (4 * r["tauD_flcs"])).cpu().numpy()
    D_in = sp["D"].cpu().numpy()
    D_all = (w0**2 / (4 * r["tauD_all"])).cpu().numpy()

    fig = plt.figure(figsize=(10, 4.3))
    grid = fig.add_gridspec(2, 3, width_ratios=[0.62, 1.15, 1.3], height_ratios=[1, 1])
    ax_m = fig.add_subplot(grid[:, 0])
    ax_t = fig.add_subplot(grid[0, 1])
    ax_h = fig.add_subplot(grid[1, 1])
    ax_g = fig.add_subplot(grid[:, 2])

    # --- left: x–z slice through the focus ---
    half_x, half_z = 1.1, 0.95 * r["box"][2].item() / 2
    xs, zs = np.linspace(-half_x, half_x, 60), np.linspace(-half_z, half_z, 120)
    X, Z = np.meshgrid(xs, zs)
    psf = np.exp(-2 * X**2 / w0**2 - 2 * Z**2 / z0**2)
    ax_m.imshow(psf, extent=(-half_x, half_x, -half_z, half_z), origin="lower", cmap="Greens", vmin=0, vmax=2.5)
    ax_m.add_patch(Ellipse((0, 0), 2 * w0, 2 * z0, fill=False, ec="#16a34a", lw=1.2, ls="--"))
    ax_m.set_xlim(-half_x, half_x)
    ax_m.set_ylim(-half_z, half_z)
    ax_m.set_aspect("equal")
    ax_m.grid(False)
    ax_m.set_xlabel("x (µm)")
    ax_m.set_ylabel("z (µm)", labelpad=0)
    ax_m.set_title("confocal volume", fontsize=10)
    dots = ax_m.scatter([], [], s=[], edgecolors="none")
    clock = ax_m.text(0.04, 0.97, "", transform=ax_m.transAxes, fontsize=8, va="top")

    # --- middle top: photons per 50 µs ---
    t_ms = np.arange(len(fine)) * frame_dt * 1e3
    ax_t.set_xlim(0, t_ms[-1] + frame_dt * 1e3)
    ax_t.set_ylim(0, max(4, fine.sum(1).max() * 1.15))
    ax_t.set_xlabel("time (ms)", labelpad=1)
    ax_t.set_ylabel(f"photons / {frame_dt * 1e6:.0f} µs", fontsize=8)
    ax_t.set_title("photon trace (bursts = crossings)", fontsize=10)
    for k in range(n_s):
        ax_t.plot([], [], color=colors[k], lw=4, label=sp["names"][k])
    ax_t.plot([], [], color="#9ca3af", lw=4, label="background")
    ax_t.legend(fontsize=6.5, loc="upper right", ncol=3, handlelength=1, columnspacing=0.8)
    stack_cols = colors + ["#9ca3af"]

    # --- middle bottom: TCSPC histogram ---
    ax_h.set_yscale("log")
    ax_h.set_xlim(0, 20)
    y_lo, y_hi = max(1, h_snap[-1].min() * 0.5), h_snap[-1].max() * 2
    ax_h.set_ylim(y_lo, y_hi)
    _plain_log_ticks(ax_h.yaxis, y_lo, y_hi)
    ax_h.set_xlabel("microtime after laser pulse (ns)", labelpad=1)
    ax_h.set_ylabel("photons", fontsize=8)
    hist_line = ax_h.plot([], [], ".", ms=2, color="#374151")[0]
    fit_lines = [ax_h.plot([], [], "--", lw=1, color=colors[k])[0] for k in range(n_s)]
    total_fit = ax_h.plot([], [], "-", lw=1, color="#111827")[0]
    tau_txt = ax_h.text(0.97, 0.93, "", transform=ax_h.transAxes, fontsize=7.5, ha="right", va="top")

    # --- right: G(τ) ---
    ax_g.set_xscale("log")
    ax_g.set_xlim(lag[0], lag[-1])
    _plain_log_ticks(ax_g.xaxis, lag[0], lag[-1])
    top = max(G_snap[-1][:, :3].max(), 0.1)
    ax_g.set_ylim(-0.08 * top, 1.25 * top)
    ax_g.set_xlabel("lag τ (ms)")
    ax_g.set_ylabel("G(τ)")
    pts_all = ax_g.plot([], [], "o", ms=2.2, color="#374151", label="all photons (FCS)")[0]
    pts_k = [ax_g.plot([], [], "o", ms=2.2, color=colors[k], label=f"FLCS {sp['names'][k]}")[0] for k in range(n_s)]
    line_all = ax_g.plot([], [], "-", lw=1.2, color="#111827")[0]
    line_k = [ax_g.plot([], [], "-", lw=1.5, color=colors[k])[0] for k in range(n_s)]
    ax_g.legend(fontsize=7, loc="upper right")
    acc_txt = ax_g.text(0.03, 0.04, "", transform=ax_g.transAxes, fontsize=8)
    fit_txt = [
        ax_g.text(0.97, 0.6 - 0.07 * k, "", transform=ax_g.transAxes, fontsize=7.5, ha="right", color=c)
        for k, c in enumerate(colors + ["#374151"])
    ]

    def update(j: int):
        # molecules in the slice, bigger and brighter near the focus
        p, vis = pos[j], np.abs(pos[j][:, 1]) < SLICE
        dots.set_offsets(p[vis][:, [0, 2]])
        dots.set_sizes(6 + 60 * W[j][vis])
        dots.set_color([colors[s] for s in who[vis]])
        clock.set_text(f"t = {t_ms[j]:.2f} ms")
        # photon trace so far, stacked by source
        for coll in list(ax_t.collections):
            coll.remove()
        base = np.zeros(j + 1)
        for k in range(n_s + 1):
            top_k = base + fine[: j + 1, k]
            ax_t.fill_between(t_ms[: j + 1], base, top_k, step="post", color=stack_cols[k], lw=0)
            base = top_k
        # histogram and correlations from the first part of the long trace
        snap = min(j * n_snap // n_build, n_snap - 1)
        hist_line.set_data(t_ns, h_snap[snap])
        pts_all.set_data(lag, G_snap[snap][0])
        for k in range(n_s):
            pts_k[k].set_data(lag, G_snap[snap][1 + k])
        acc_txt.set_text(f"correlated: first {r['fractions'][snap] * r['duration']:.2f} s of {r['duration']:.1f} s")
        if j >= n_build - 1:  # the whole trace is in: show the fits
            total_fit.set_data(t_ns, flim_fit)
            for k in range(n_s):
                fit_lines[k].set_data(t_ns, comps[:, k])
                line_k[k].set_data(lag, fit_flcs[:, k])
            line_all.set_data(lag, fit_all)
            tau_txt.set_text("  ".join(f"τ = {v:.2f} ns" for v in r["tau_fl"].tolist()))
            for k in range(n_s):
                fit_txt[k].set_text(f"FLCS: D = {D_flcs[k]:.0f} µm²/s (input {D_in[k]:.0f})")
            fit_txt[n_s].set_text("plain FCS fit: D = " + " / ".join(f"{d:.0f}" for d in D_all) + " µm²/s")

    frames = frame_indices(n_frames, n_frames)
    fig.tight_layout(pad=0.6)
    return save_gif(fig, update, frames, path)


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _plain_log_ticks(axis, lo: float, hi: float) -> None:
    """Decade ticks with plain-text labels and no minor ticks: log axes otherwise re-typeset mathtext every frame."""
    decades = range(int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi))) + 1)
    axis.set_ticks([10.0**d for d in decades], [f"10{str(d).translate(SUPERSCRIPT)}" for d in decades])
    axis.set_minor_locator(plt.NullLocator())
