"""Animate the fit: the Hubble diagram relative to an empty universe, and the path through the Ω_m–Ω_Λ plane."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import equation, tensors

LEVELS = [2.30, 6.18, 11.83]  # Δχ² of the 1σ, 2σ, 3σ regions for two parameters


def binned(z, mu, sigma, n_bins: int = 16):
    """Inverse-variance mean of the residuals in log-spaced redshift bins (for the eye only, not for the fit)."""
    edges = np.geomspace(z.min(), z.max() * 1.001, n_bins + 1)
    k = np.digitize(z, edges) - 1
    w = sigma**-2
    sw = np.bincount(k, w, n_bins)
    keep = sw > 0
    zc = np.bincount(k, w * z, n_bins)[keep] / sw[keep]
    mc = np.bincount(k, w * mu, n_bins)[keep] / sw[keep]
    return zc, mc, sw[keep] ** -0.5


def animate(z, mu, sigma, fit, Om, OL, chi2, sig, path: Path) -> Path:
    zs = torch.logspace(np.log10(float(z.min())), np.log10(float(z.max())), 200, dtype=z.dtype)
    nodes = tensors.nodes_for(z)
    H0 = fit["best"][0]

    def mu_of(zz, h0, om, ol):
        return equation.distance_modulus(equation.luminosity_distance(zz, nodes, h0, om, ol)).numpy()

    empty_data = mu_of(z, H0, 0.0, 0.0)
    empty_curve = mu_of(zs, H0, 0.0, 0.0)
    res = (mu - torch.as_tensor(empty_data)).numpy()
    zn, sn = z.numpy(), sigma.numpy()
    zb, mb, eb = binned(zn, res, sn)
    path_p = fit["path"].numpy()  # (S, 3)
    curves = np.stack([mu_of(zs, *p) for p in fit["path"][frame_indices(len(path_p), 70)]])
    frames = frame_indices(len(path_p), 70)
    curve_of = dict(zip(frames, curves))
    eds = mu_of(zs, H0, 1.0, 0.0) - empty_curve
    best = mu_of(zs, *fit["best"]) - empty_curve

    fig, (ax, ax_c) = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax.scatter(zn, res, s=3, color="#d1d5db", lw=0, label=f"{len(zn)} SNe Ia (Pantheon+)")
    ax.errorbar(zb, mb, eb, fmt="o", ms=4, color="#374151", lw=1, capsize=0, label="binned mean")
    ax.axhline(0, color="#6b7280", lw=1, ls=":", label="empty universe (coasting)")
    ax.plot(zs.numpy(), eds, color=ACCENT[1], lw=1.5, label=r"matter only $\Omega_m = 1$ (decelerating)")
    ax.plot(zs.numpy(), best, color="#111827", lw=1.2, ls="--", label="best fit")
    current = ax.plot([], [], color=ACCENT[0], lw=2.2, label="Adam iterate")[0]
    ax.set_xscale("log")
    ax.set_ylim(-0.8, 0.6)
    ax.set_xlabel("redshift z")
    ax.set_ylabel(r"$\mu - \mu_\mathrm{empty}$ (mag)")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("SNe farther (fainter) than in a coasting universe ⇒ acceleration", fontsize=8.5)

    om, ol = Om.numpy(), OL.numpy()
    dchi = (chi2 - chi2.min()).numpy().T  # (B, A) for contourf
    dchi = np.where(np.isfinite(dchi), dchi, 1e9)
    ax_c.contourf(om, ol, dchi, levels=[0, *LEVELS], colors=["#1d4ed8", "#60a5fa", "#bfdbfe"])
    ax_c.contour(om, ol, dchi, levels=LEVELS, colors="#1e40af", linewidths=0.6)
    invalid = ~np.isfinite((chi2 - chi2.min()).numpy().T)
    ax_c.contourf(om, ol, invalid.astype(float), levels=[0.5, 1.5], colors=["#e5e7eb"])
    ax_c.text(0.05, 1.45, "no Big Bang", fontsize=7, color="#6b7280")
    ax_c.plot(om, 1 - om, color="#374151", lw=1, ls="--")
    ax_c.text(0.13, 0.9, "flat", fontsize=7, color="#374151", rotation=-38)
    ax_c.plot(om, om / 2, color=ACCENT[3], lw=1)
    ax_c.text(0.75, 0.43, "q₀ = 0", fontsize=7, color=ACCENT[3], rotation=20)
    ax_c.text(0.75, 1.2, "accelerating", fontsize=7, color="#374151")
    ax_c.text(0.9, 0.25, "decelerating", fontsize=7, color="#374151")
    ax_c.axhline(0, color="#9ca3af", lw=0.8, ls=":")
    trail = ax_c.plot([], [], color=ACCENT[4], lw=1.2)[0]
    dot = ax_c.plot([], [], "o", color=ACCENT[4], ms=6, mec="white")[0]
    ax_c.set_xlim(om[0], om[-1])
    ax_c.set_ylim(ol[0], ol[-1])
    ax_c.set_xlabel(r"$\Omega_m$")
    ax_c.set_ylabel(r"$\Omega_\Lambda$")
    ax_c.set_title(f"1–3σ from the χ² grid · Ω_Λ ≤ 0 excluded at {sig['sigma_lambda']:.1f}σ", fontsize=8.5)
    label = ax_c.text(0.98, 0.02, "", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=7.5)

    def update(step: int):
        p = path_p[step]
        current.set_data(zs.numpy(), curve_of[step] - empty_curve)
        trail.set_data(path_p[: step + 1, 1], path_p[: step + 1, 2])
        dot.set_data([p[1]], [p[2]])
        q0 = p[1] / 2 - p[2]
        label.set_text(
            f"step {step}: χ² = {fit['chi2'][step]:.0f}\n"
            f"H₀ ={p[0]:.1f}, Ω_m = {p[1]:.2f}, Ω_Λ = {p[2]:.2f}, q₀ = {q0:+.2f}"
        )

    fig.tight_layout()
    return save_gif(fig, update, frames, path)
