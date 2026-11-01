"""Watch power iteration converge: flux shape, k per iteration, and k_eff against slab width."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(
    x: torch.Tensor,
    width: float,
    ks: torch.Tensor,
    fluxes: torch.Tensor,
    k_exact: float,
    widths: torch.Tensor,
    k_widths: torch.Tensor,
    l_crit: float,
    path: Path,
) -> Path:
    x_np, phi = x.cpu().numpy(), fluxes.cpu().numpy()  # (N,), (I+1, N)
    k_np = ks.cpu().numpy()
    it = np.arange(len(k_np))

    fig, (ax_f, ax_k, ax_w) = plt.subplots(1, 3, figsize=(10.5, 3.6))
    ax_f.plot(x_np, np.sin(np.pi * x_np / width), color="#9ca3af", lw=2, ls="--", label=r"$\sin(\pi x/L)$")
    (start,) = ax_f.plot(x_np, phi[0], color="#93c5fd", lw=1, label="start guess")
    (line,) = ax_f.plot([], [], color="#2563eb", lw=2, label="iterate")
    ax_f.set_xlim(0, width)
    ax_f.set_ylim(0, 1.15)
    ax_f.set_xlabel("x (cm)")
    ax_f.set_ylabel("flux φ (peak = 1)")
    ax_f.legend(loc="lower center", fontsize=7)
    title = ax_f.set_title("")

    ax_k.axhline(k_exact, color="#9ca3af", lw=2, ls="--", label="analytic k")
    (k_line,) = ax_k.plot([], [], "o-", color="#dc2626", ms=3, lw=1, label="power iteration")
    ax_k.set_xlim(0, it[-1])
    lo, hi = min(k_np.min(), k_exact), max(k_np.max(), k_exact)
    ax_k.set_ylim(lo - 0.1 * (hi - lo) - 1e-3, hi + 0.1 * (hi - lo) + 1e-3)
    ax_k.set_xlabel("iteration n")
    ax_k.set_ylabel("$k^{(n)}$")
    ax_k.legend(loc="upper right", fontsize=7)

    w_np, kw_np = widths.cpu().numpy(), k_widths.cpu().numpy()
    ax_w.plot(w_np, kw_np, color="#2563eb", lw=2)
    ax_w.axhline(1.0, color="#9ca3af", lw=1)
    if np.isfinite(l_crit) and w_np[0] <= l_crit <= w_np[-1]:
        ax_w.axvline(l_crit, color="#f59e0b", lw=1, ls="--")
        ax_w.text(l_crit, kw_np.min(), f" critical\n L = {l_crit:.1f} cm", color="#b45309", fontsize=8, va="bottom")
    ax_w.plot([width], [k_exact], "o", color="#dc2626")
    ax_w.set_xlabel("slab width L (cm)")
    ax_w.set_ylabel("$k_{eff}$")
    ax_w.set_title("eigen solve, batched over L", fontsize=9)

    def update(step: int):
        line.set_data(x_np, phi[step])
        start.set_alpha(1.0 if step < 3 else 0.4)
        k_line.set_data(it[: step + 1], k_np[: step + 1])
        title.set_text(f"iteration {step}:  k = {k_np[step]:.5f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(k_np), 61), path, fps=10)
