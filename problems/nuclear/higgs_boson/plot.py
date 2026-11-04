"""The histogram fills up, the fit walks to the peak, then the local significance is scanned against mass.

Left: m_γγ spectrum with the fit, data minus background, local p₀ against mass. Right: the Mexican-hat potential
and the background-only toys (q₀ at 125 GeV against ½χ²₁, and the largest q₀ anywhere: the look-elsewhere effect).
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from . import equation, tensors

BLUE, RED, GREY, AMBER, GREEN = "#2563eb", "#dc2626", "#9ca3af", "#f59e0b", "#16a34a"
N_FILL, N_FIT, N_SCAN = 22, 16, 22


def _model(row: torch.Tensor, x, edges, widths, sigma: float, n_sm: float):
    """Background and signal+background (per bin) for one path row (m_H, μ, θ...)."""
    b = widths * equation.background(row[2:], x)
    s = row[1] * n_sm * equation.signal_shape(edges, row[0], sigma)
    return b.numpy(), (b + s).numpy()


def animate(data, x, sigma, n_sm, fit, masses, q0_obs, q0_toys, k125, lee, potential, path: Path) -> Path:
    data = {k: v.cpu() for k, v in data.items()}
    x = x.cpu()
    c, counts, widths, edges = data["centers"].numpy(), data["counts"], data["widths"], data["edges"]
    fractions = torch.linspace(0.03, 1.0, N_FILL, dtype=torch.float64)
    filling = tensors.thinned(counts, fractions, seed=1).numpy()  # (J, N)
    steps = fit["path"]
    pick = np.unique(np.geomspace(1, len(steps), N_FIT).astype(int) - 1)
    models = [_model(steps[i], x, edges, widths, sigma, n_sm) for i in pick]
    b_final, sb_final = _model(steps[-1], x, edges, widths, sigma, n_sm)
    m, p0 = masses.cpu().numpy(), equation.p_value(q0_obs.cpu()).numpy()
    q_toys = q0_toys.cpu()

    fig = plt.figure(figsize=(10, 6.6))
    grid = fig.add_gridspec(3, 2, width_ratios=[2.1, 1], height_ratios=[2, 1, 1.3])
    ax_h = fig.add_subplot(grid[0, 0])
    ax_r = fig.add_subplot(grid[1, 0], sharex=ax_h)
    ax_p = fig.add_subplot(grid[2, 0], sharex=ax_h)
    ax_v = fig.add_subplot(grid[0, 1])
    ax_t = fig.add_subplot(grid[1:, 1])

    scale = counts.max().item() * 1.12
    ax_h.set_ylim(0, scale)
    ax_h.set_xlim(edges[0].item(), edges[-1].item())
    ax_h.set_ylabel(f"events / {widths[0].item():g} GeV")
    pts = ax_h.errorbar(c, filling[0], yerr=np.sqrt(filling[0]), fmt="o", ms=2, color="#111827", lw=0.8)
    (line_b,) = ax_h.plot([], [], color=BLUE, lw=1.5, ls="--", label="background")
    (line_sb,) = ax_h.plot([], [], color=RED, lw=2, label="signal + background")
    ax_h.legend(loc="upper right", fontsize=8)
    title = ax_h.set_title("", fontsize=10)
    info = ax_h.text(0.02, 0.05, "", transform=ax_h.transAxes, fontsize=8)

    resid = counts.numpy() - b_final
    ax_r.axhline(0, color=BLUE, lw=1, ls="--")
    ax_r.set_ylim(-4.5 * math.sqrt(b_final.max()), 4.5 * math.sqrt(b_final.max()))
    ax_r.set_ylabel("data − bkg")
    resid_pts = ax_r.errorbar(c, resid * 0, yerr=np.sqrt(counts.numpy()), fmt="o", ms=2, color="#111827", lw=0.6)
    (line_r,) = ax_r.plot([], [], color=RED, lw=2)
    resid_pts[0].set_visible(False)
    for bar in resid_pts[2]:
        bar.set_visible(False)

    ax_p.set_yscale("log")
    ax_p.set_ylim(min(p0.min(), 1e-3) / 5, 1.0)
    ax_p.set_ylabel("local $p_0$")
    ax_p.set_xlabel(r"diphoton mass $m_{\gamma\gamma}$ (GeV)")
    for z in (1, 2, 3):
        level = 0.5 * math.erfc(z / math.sqrt(2))
        ax_p.axhline(level, color=GREY, lw=0.8, ls=":")
        ax_p.text(edges[-1].item() - 0.5, level, f"{z}σ", fontsize=7, ha="right", va="bottom", color="#6b7280")
    ax_p.axvline(tensors.M_H_PDG[0], color=AMBER, lw=1)
    (line_p,) = ax_p.plot([], [], color=BLUE, lw=2)
    plt.setp(ax_h.get_xticklabels(), visible=False)
    plt.setp(ax_r.get_xticklabels(), visible=False)

    # Mexican hat: V along rays φ = r e^{iα} of the complex plane (all phases give the same curve)
    mu2, lam = potential
    r = torch.linspace(-265, 265, 401, dtype=torch.float64)
    alpha = torch.tensor([0.0, 1.0, 2.5], dtype=torch.float64)
    phi = r[None, :] * torch.exp(1j * alpha[:, None])  # (3, R) complex
    pot = equation.mexican_hat(phi, mu2, lam).numpy() / 1e8  # (3, R) in 10⁸ GeV⁴
    vac = tensors.V_EW / math.sqrt(2)
    v_min = -lam * tensors.V_EW**4 / 4 / 1e8
    for k, ls in enumerate(["-", "--", ":"]):
        ax_v.plot(r, pot[k], color=BLUE, lw=2 if k == 0 else 1.2, ls=ls)
    h = np.linspace(-60, 60, 50)  # φ = (v + h)/√2: V ≈ V_min + ½ m_H² h²
    m_h2 = 2 * lam * tensors.V_EW**2
    for sign in (-1, 1):
        ax_v.plot(sign * (vac + h / math.sqrt(2)), v_min + 0.5 * m_h2 * h**2 / 1e8, color=RED, lw=1.5)
    ax_v.plot([-vac, vac], [v_min, v_min], "o", color=RED, ms=4)
    ax_v.plot(0, 0, "o", color=AMBER, ms=4)
    ax_v.set_ylim(1.3 * v_min, 0.9)
    ax_v.set_xlabel(r"$r$ along $\phi = r\,e^{i\alpha}$ (GeV)", fontsize=8)
    ax_v.set_ylabel(r"$V$ ($10^8$ GeV$^4$)", fontsize=8)
    ax_v.tick_params(labelsize=7)
    ax_v.set_title(
        rf"$V=-\mu^2|\phi|^2+\lambda|\phi|^4$, $\lambda$ = {lam:.3f}"
        + "\n"
        + rf"minimum $|\phi| = v/\sqrt{{2}}$ = {vac:.0f} GeV; red: $\frac{{1}}{{2}}m_H^2h^2$",
        fontsize=8,
    )

    # toys: q₀ at 125 GeV and the largest q₀ anywhere
    bins = np.linspace(0, 14, 29)
    q125 = q_toys[:, k125].numpy()
    ax_t.hist(q125[q125 > 0], bins=bins, color=BLUE, alpha=0.5, label=r"toys: $q_0$ at 125 GeV")
    ax_t.hist(lee["q_max"].numpy(), bins=bins, color=AMBER, alpha=0.5, label=r"toys: max $q_0$ over mass")
    grid_q = np.linspace(0.05, 14, 200)
    chi2 = np.exp(-grid_q / 2) / np.sqrt(2 * np.pi * grid_q)
    width = bins[1] - bins[0]
    ax_t.plot(grid_q, 0.5 * chi2 * len(q125) * width, color="#111827", lw=1.2, label=r"$\frac{1}{2}\chi^2_1$ (Wilks)")
    obs_line = ax_t.axvline(0, color=RED, lw=2, label=r"observed $q_0$(125)", visible=False)
    ax_t.set_xlim(0, bins[-1])
    ax_t.set_yscale("log")
    ax_t.set_ylim(0.5, len(q125))
    ax_t.set_xlabel(r"$q_0 = -2\ln\lambda(0)$")
    ax_t.legend(fontsize=7, loc="upper right")
    lee_text = ax_t.text(0.04, 0.04, "", transform=ax_t.transAxes, fontsize=7.5)

    def update(frame: int):
        if frame < N_FILL:
            f = fractions[frame].item()
            y = filling[frame]
            _set_errorbar(pts, c, y, np.sqrt(y))
            title.set_text(f"ATLAS Open Data, 13 TeV: {10 * f:.1f} fb⁻¹ collected")
            return
        _set_errorbar(pts, c, counts.numpy(), np.sqrt(counts.numpy()))
        if frame < N_FILL + N_FIT:
            j = min(frame - N_FILL, len(models) - 1)
            b, sb = models[j]
            row = steps[pick[j]]
            line_b.set_data(c, b)
            line_sb.set_data(c, sb)
            title.set_text(f"fitting with autograd: m_H = {row[0].item():.1f} GeV, μ = {row[1].item():.2f}")
            return
        line_b.set_data(c, b_final)
        line_sb.set_data(c, sb_final)
        resid_pts[0].set_visible(True)
        for bar in resid_pts[2]:
            bar.set_visible(True)
        _set_errorbar(resid_pts, c, resid, np.sqrt(counts.numpy()))
        line_r.set_data(c, sb_final - b_final)
        title.set_text("H → γγ: fit to 10 fb⁻¹ of real ATLAS data")
        info.set_text(
            f"$m_H$ = {fit['m_h']:.1f} ± {fit['m_err']:.1f} GeV,  μ = {fit['mu']:.2f} ± {fit['mu_err']:.2f},  "
            f"local Z = {fit['z']:.1f}σ"
        )
        k = int(round((frame - N_FILL - N_FIT + 1) / N_SCAN * len(m)))
        line_p.set_data(m[:k], p0[:k])
        if frame == N_FILL + N_FIT + N_SCAN - 1:
            obs_line.set_xdata([q0_obs[k125].item()])
            obs_line.set_visible(True)
            lee_text.set_text(
                f"local p₀(125) = {lee['p_local']:.1e}\nglobal p ≈ {lee['p_global_gv']:.1e} (Gross–Vitells)\n"
                f"{lee['p_global_toys']:.0%} of toys beat it somewhere"
            )

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(N_FILL + N_FIT + N_SCAN, N_FILL + N_FIT + N_SCAN), path, fps=12)


def _set_errorbar(container, x, y, yerr) -> None:
    """Move the points and vertical error bars of an errorbar() container."""
    line, _, (bars,) = container
    line.set_data(x, y)
    bars.set_segments([np.array([[xi, yi - ei], [xi, yi + ei]]) for xi, yi, ei in zip(x, y, yerr)])
