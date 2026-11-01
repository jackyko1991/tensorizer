"""Animate the fit: data with error bars, the model and its ±σ_tot error band, normalised residuals, χ² vs iteration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import model, total_sigma


def animate(t, x, sigma, history: torch.Tensor, chi2_hist: torch.Tensor, jitter: bool, path: Path) -> Path:
    tn, xn, sn = (a.cpu().numpy() for a in (t, x, sigma))
    hist = history.cpu()
    K = len(hist)
    chi = chi2_hist.cpu().numpy()
    fine = torch.linspace(0, tn[-1], 400, dtype=hist.dtype)
    sig_fine = torch.from_numpy(np.interp(fine.numpy(), tn, sn))  # error bars interpolated onto the fine grid
    curves = np.stack([model(fine, p).numpy() for p in hist])  # (K, 400)
    at_data = np.stack([model(t.cpu(), p).numpy() for p in hist])  # (K, N)
    if jitter:
        s_fine = np.stack([total_sigma(sig_fine, p[5]).numpy() for p in hist])
        s_data = np.stack([total_sigma(sigma.cpu(), p[5]).numpy() for p in hist])
    else:
        s_fine = np.tile(sig_fine.numpy(), (K, 1))
        s_data = np.tile(sn, (K, 1))
    resid = (xn - at_data) / s_data  # (K, N) normalised residuals r_i
    fn = fine.numpy()
    dof = len(tn) - (6 if jitter else 5)

    fig = plt.figure(figsize=(8.2, 4.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[1.6, 1])
    ax_d = fig.add_subplot(grid[0, 0])
    ax_r = fig.add_subplot(grid[1, 0], sharex=ax_d)
    ax_c = fig.add_subplot(grid[0, 1])
    ax_h = fig.add_subplot(grid[1, 1])

    # --- data with error bars, model curve and its ±σ_tot band ---
    ax_d.errorbar(tn, xn, yerr=sn, fmt="o", ms=2.5, color="#111827", ecolor="#6b7280", elinewidth=0.8, capsize=0)
    (curve,) = ax_d.plot([], [], color="#dc2626", lw=1.6)
    band = [ax_d.fill_between(fn, curves[0] - s_fine[0], curves[0] + s_fine[0], color="#f87171", alpha=0.35, lw=0)]
    ax_d.set_ylim(xn.min() - 3, xn.max() + 3)
    ax_d.set_ylabel("x (mm)")
    ax_d.set_title(r"data $\pm\sigma_i$, model $f(t;\theta)$ $\pm\sigma_{tot}$", fontsize=10)
    ax_d.tick_params(labelbottom=False)
    label = ax_d.text(
        0.98,
        0.95,
        "",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#e5e7eb", alpha=0.9),
    )

    # --- normalised residuals r_i = (x_i - f_i)/σ_tot,i ---
    ax_r.axhspan(-1, 1, color="#e5e7eb", lw=0)
    ax_r.axhline(0, color="#9ca3af", lw=1)
    (res_pts,) = ax_r.plot([], [], "o", ms=2.5, color="#2563eb")
    ax_r.set_ylim(-4, 4)
    ax_r.set_xlim(0, tn[-1])
    ax_r.set_xlabel("t (s)")
    ax_r.set_ylabel(r"$r_i$")

    # --- χ² vs iteration ---
    it = np.arange(K)
    ax_c.set_yscale("log")
    ax_c.set_xlim(0, K - 1)
    ax_c.set_ylim(min(chi.min(), dof) * 0.7, chi.max() * 1.5)
    ax_c.axhline(dof, color="#16a34a", ls="--", lw=1)
    ax_c.text(0.97 * (K - 1), dof * 1.12, f"dof = {dof}", fontsize=7.5, color="#16a34a", ha="right")
    (chi_line,) = ax_c.plot([], [], color="#2563eb", lw=1.4)
    ax_c.set_xlabel("iteration", labelpad=1)
    ax_c.set_title(r"$\chi^2=\sum_i r_i^2$ (last: L-BFGS)", fontsize=10)

    # --- histogram of normalised residuals vs N(0, 1) ---
    edges = np.linspace(-4, 4, 13)
    width = edges[1] - edges[0]
    centers = 0.5 * (edges[1:] + edges[:-1])
    bars = ax_h.bar(centers, np.zeros_like(centers), width=width * 0.9, color="#93c5fd", edgecolor="#2563eb", lw=0.5)
    g = np.linspace(-4, 4, 200)
    ax_h.plot(g, np.exp(-(g**2) / 2) / np.sqrt(2 * np.pi), color="#dc2626", lw=1.4)
    ax_h.set_xlim(-4, 4)
    ax_h.set_ylim(0, 0.75)
    ax_h.set_xlabel(r"$r_i$ vs $N(0,1)$", labelpad=1)

    def update(k: int):
        curve.set_data(fn, curves[k])
        band[0].remove()
        band[0] = ax_d.fill_between(fn, curves[k] - s_fine[k], curves[k] + s_fine[k], color="#f87171", alpha=0.35, lw=0)
        p = hist[k].numpy()
        stage = "L-BFGS" if k == K - 1 else f"Adam {k}"
        s_txt = f"  s={np.exp(p[5]):.2f}" if jitter else ""
        label.set_text(f"{stage}: A={p[0]:.1f} γ={p[1]:.3f} ω={p[2]:.3f}{s_txt}\nχ²/dof = {chi[k] / dof:.2f}")
        res_pts.set_data(tn, resid[k])
        chi_line.set_data(it[: k + 1], chi[: k + 1])
        counts = np.histogram(np.clip(resid[k], -3.99, 3.99), bins=edges)[0] / (len(tn) * width)
        for bar, h in zip(bars, counts):
            bar.set_height(h)

    fig.tight_layout()
    frames = frame_indices(K - 1, 63) + [K - 1]  # Adam steps, then the polished L-BFGS result
    frames += [K - 1] * 10  # hold the final fit for half a second
    return save_gif(fig, update, frames, path)
