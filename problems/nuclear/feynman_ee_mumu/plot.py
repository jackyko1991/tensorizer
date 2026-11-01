"""Sweep the collision energy: angular distribution as a polar shape, total cross section against √s."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

CB, CO, CG = "#2563eb", "#f59e0b", "#9ca3af"


def animate(
    sqrt_s: torch.Tensor,
    theta: torch.Tensor,
    dsdo: torch.Tensor,
    sigma: torch.Tensor,
    exact: torch.Tensor,
    labels: tuple[str, str, float],
    path: Path,
) -> Path:
    """dsdo (E, B) nb/sr, sigma and exact (E,) nb."""
    title, sym, threshold = labels
    e, th = sqrt_s.cpu().numpy(), theta.cpu().numpy()
    shape = dsdo.cpu().numpy()
    shape = shape / shape[:, len(th) // 2 : len(th) // 2 + 1]  # normalise to 1 at θ = 90°
    sig, ex = sigma.cpu().numpy(), exact.cpu().numpy()
    full = np.concatenate([th, 2 * np.pi - th[::-1]])  # mirror: the distribution is symmetric about the beam

    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax_p.set_xlim(-2.3, 2.3)
    ax_p.set_ylim(-2.3, 2.3)
    ax_p.set_aspect("equal")
    ax_p.axhline(0, color="#374151", lw=0.8)
    ax_p.annotate("", xy=(2.2, 0), xytext=(1.4, 0), arrowprops={"arrowstyle": "->", "color": "#374151"})
    ax_p.text(1.45, 0.12, "e⁻ beam", fontsize=8)
    ref = 1 + np.cos(full) ** 2
    ax_p.plot(ref * np.cos(full), ref * np.sin(full), ls="--", color=CO, lw=1.5, label="1 + cos²θ  (√s ≫ 2m)")
    ax_p.plot(np.cos(full), np.sin(full), ls=":", color=CG, lw=1.2, label="isotropic  (threshold)")
    (curve,) = ax_p.plot([], [], color=CB, lw=2.2, label="traces")
    ax_p.legend(loc="lower center", fontsize=7, frameon=False, ncols=1, bbox_to_anchor=(0.5, -0.02))
    ax_p.set_title(f"dσ/dΩ shape for {sym}⁺{sym}⁻ (1 at 90°)", fontsize=10)
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    readout = ax_p.text(0.02, 0.97, "", transform=ax_p.transAxes, va="top", fontsize=9, family="monospace")

    ax_s.loglog(e, ex, color=CO, lw=3, alpha=0.5, label="4πα²β(1+2m²/s)/3s")
    ax_s.loglog(e, 86.86 / e**2, color=CG, ls=":", lw=1.2, label="4πα²/3s  (massless)")
    ax_s.loglog(e, sig, color=CB, lw=1.5, label="∫ traces dΩ")
    ax_s.axvline(threshold, color="#dc2626", lw=0.8, ls="--")
    ax_s.text(threshold * 1.05, sig.max() * 0.4, f"2m_{sym}", color="#dc2626", fontsize=8)
    (dot,) = ax_s.plot([], [], "o", color=CB, ms=7)
    ax_s.set_xlabel("√s (GeV)")
    ax_s.set_ylabel("σ (nb)")
    ax_s.set_title(title, fontsize=10)
    ax_s.legend(fontsize=7, frameon=False)
    ax_s.grid(True, which="both", alpha=0.2)

    def update(j: int):
        r = np.concatenate([shape[j], shape[j][::-1]])
        curve.set_data(r * np.cos(full), r * np.sin(full))
        dot.set_data([e[j]], [sig[j]])
        readout.set_text(f"√s = {e[j]:.3f} GeV\nσ  = {sig[j]:.3g} nb")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(e), 90), path, fps=12)
