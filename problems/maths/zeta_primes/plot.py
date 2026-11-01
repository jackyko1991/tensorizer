"""Z(t) with its zeros, π(x) against the classical approximations, and π(x) rebuilt from more and more zeros."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    t: torch.Tensor,
    z: torch.Tensor,
    gammas: torch.Tensor,
    x_log: torch.Tensor,
    pi_log: torch.Tensor,
    approx: dict,
    x_lin: torch.Tensor,
    exact: torch.Tensor,
    pi_k: torch.Tensor,
    path: Path,
) -> Path:
    t, z, g = t.cpu().numpy(), z.cpu().numpy(), gammas.cpu().numpy()
    fig, (ax_z, ax_p, ax_e) = plt.subplots(1, 3, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.1, 1, 1.3]})

    show = t <= 60
    ax_z.plot(t[show], z[show], color="#374151", lw=1)
    ax_z.axhline(0, color="#9ca3af", lw=0.8)
    g60 = g[g <= 60]
    used = ax_z.scatter(g60, np.zeros_like(g60), s=22, zorder=3, color="#9ca3af")
    ax_z.set_xlabel("$t$")
    ax_z.set_title(r"$Z(t) = e^{i\theta(t)}\zeta(1/2 + it)$", fontsize=9)

    x, p = x_log.cpu().numpy(), pi_log.cpu().numpy().astype(float)
    for (name, a), colour in zip(approx.items(), ACCENT[1:]):
        ax_p.loglog(x, np.abs(a.cpu().numpy() / p - 1), color=colour, lw=1.3, label=name)
    ax_p.set_xlabel("$x$")
    ax_p.set_title(r"relative error $|\mathrm{approx}/\pi(x) - 1|$", fontsize=9)
    ax_p.legend(fontsize=7, loc="lower left")

    xl, ex, pk = x_lin.cpu().numpy(), exact.cpu().numpy(), pi_k.cpu().numpy()
    ax_e.step(xl, ex, where="post", color="#374151", lw=1.2, label=r"$\pi(x)$ (sieve)")
    ax_e.plot(xl, pk[0], color="#9ca3af", lw=1, ls="--", label="no zeros (≈ R(x))")
    (curve,) = ax_e.plot(xl, pk[0], color="#2563eb", lw=1.3, label="with zeros")
    ax_e.set_xlabel("$x$")
    ax_e.set_ylim(-1, ex.max() * 1.1 + 1)
    ax_e.legend(fontsize=7, loc="upper left")
    title = ax_e.set_title("", fontsize=9)

    n_max = len(g)
    counts = sorted({round(n_max * (f / 59) ** 2) for f in range(60)})  # slow at first, faster later

    def update(f: int):
        k = counts[f]
        curve.set_ydata(pk[k])
        used.set_color(["#dc2626" if j < k else "#9ca3af" for j in range(len(g60))])
        title.set_text(f"explicit formula with the first {k} zeros")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(counts), len(counts)), path, fps=8)
