"""Animate the fit converging on the FIRAS points, then zoom into the residuals in units of 10⁻⁴ of the peak."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

GHZ_PER_CM = 29.9792458  # 1 cm⁻¹ = 29.98 GHz


def animate(view: dict, path: Path, hold: int = 6, zoom: int = 30) -> Path:
    """view: numpy arrays from main.py. The first phase steps through the fit, the second shrinks the residual axis."""
    nu, data, err = view["nu"], view["data"], view["sigma"]
    curves, T_hist, chi2_hist = view["curves"], view["T"], view["chi2"]  # (K, G), (K,), (K,)
    nu_fine = view["nu_fine"]
    model_at_data = view["model_at_data"]  # (K, F) T-only model at the data points, per shown iteration
    peak = data.max()
    res = (data[None] - model_at_data) / peak * 1e4  # (K, F) residuals in units of 10⁻⁴ of the peak
    n_iter = len(T_hist)
    schedule = [k for k in range(n_iter) for _ in range(hold)] + [n_iter - 1] * zoom
    final_span = 3.5
    start_span = max(final_span, float(np.abs(res[-1]).max()) * 1.15)

    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(8.6, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1.15 * max(peak, curves.max()))
    ax.set_xlabel("wavenumber ν̃ (cm⁻¹)")
    ax.set_ylabel("intensity I_ν (MJy/sr)")
    ax.errorbar(
        nu,
        data,
        yerr=400 * err,
        fmt="o",
        ms=3,
        color="#111827",
        ecolor="#6b7280",
        lw=1,
        label="FIRAS (error bars ×400)",
    )
    line = ax.plot([], [], "-", color=ACCENT[0], lw=2, label="Planck law, fitted T")[0]
    ax.legend(fontsize=8, loc="upper right")
    top = ax.secondary_xaxis("top", functions=(lambda c: c * GHZ_PER_CM, lambda g: g / GHZ_PER_CM))
    top.set_xlabel("frequency (GHz)", fontsize=8)
    title = fig.suptitle("", fontsize=10)

    ax_r.set_xlim(0, 23)
    ax_r.axhline(0, color="#9ca3af", lw=1)
    ax_r.set_xlabel("wavenumber ν̃ (cm⁻¹)")
    ax_r.set_ylabel("(data − blackbody) / peak  (×10⁻⁴)")
    bars = ax_r.errorbar(nu, res[0], yerr=err / peak * 1e4, fmt="o", ms=3, color="#111827", ecolor="#6b7280", lw=1)
    mu_line = ax_r.plot(nu, view["mu_limit"] / peak * 1e4, "--", color=ACCENT[1], lw=1.5, alpha=0)[0]
    y_line = ax_r.plot(nu, view["y_limit"] / peak * 1e4, "--", color=ACCENT[3], lw=1.5, alpha=0)[0]
    info = ax_r.text(0.03, 0.97, "", transform=ax_r.transAxes, va="top", fontsize=7.5, family="monospace")
    ax_r.set_title("residuals", fontsize=9)

    def update(i: int):
        k = schedule[i]
        line.set_data(nu_fine, curves[k])
        title.set_text(f"fit step {k}:  T = {T_hist[k]:.5f} K   χ²/dof = {chi2_hist[k] / view['dof']:.3g}")
        bars.lines[0].set_ydata(res[k])
        segs = [np.array([[x, r - e], [x, r + e]]) for x, r, e in zip(nu, res[k], err / peak * 1e4)]
        bars.lines[2][0].set_segments(segs)
        z = 0.0 if i < n_iter * hold else (i - n_iter * hold + 1) / zoom  # 0 → 1 during the zoom phase
        if z == 0:  # while fitting: follow the residuals
            span = max(final_span, float(np.abs(res[k]).max()) * 1.15)
        else:  # then zoom in to a few 10⁻⁴ (the last, noisiest points may leave the frame)
            span = max(final_span, start_span ** (1 - z) * final_span**z)
        ax_r.set_ylim(-span, span)
        shown = z > 0.5
        mu_line.set_alpha(0.9 if shown else 0)
        y_line.set_alpha(0.9 if shown else 0)
        mu_line.set_label(f"μ = {view['mu_lim']:.1e} (our 95 % limit)" if shown else "_")
        y_line.set_label(f"y = {view['y_lim']:.1e} (our 95 % limit)" if shown else "_")
        if shown and ax_r.get_legend() is None:
            ax_r.legend(fontsize=7, loc="lower left")
        info.set_text(view["summary"] if shown else "")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_gif(fig, update, frame_indices(len(schedule), 80), path)
