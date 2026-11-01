"""Sweep the decay angle: back-to-back momenta in the rest frame, boosted momenta in the lab."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

C1, C2, CP = "#2563eb", "#dc2626", "#374151"


def animate(
    theta: torch.Tensor,
    rest1: torch.Tensor,
    rest2: torch.Tensor,
    lab1: torch.Tensor,
    lab2: torch.Tensor,
    parent: torch.Tensor,
    m_inv: torch.Tensor,
    labels: tuple[str, str, str],
    path: Path,
) -> Path:
    """All momenta (B, 4); we draw the (z, x) components, z = flight direction of the parent."""
    deg = np.degrees(theta.cpu().numpy())
    r1, r2 = rest1.cpu().numpy()[:, [3, 1]], rest2.cpu().numpy()[:, [3, 1]]  # (B, 2) as (p_z, p_x)
    l1, l2 = lab1.cpu().numpy()[:, [3, 1]], lab2.cpu().numpy()[:, [3, 1]]
    p_par = parent.cpu().numpy()[[3, 1]]
    mass = m_inv.cpu().numpy()
    title, name1, name2 = labels

    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1, 1.6]})
    p_star = np.linalg.norm(r1[0])
    span = 1.3 * p_star
    ax_r.set_xlim(-span, span)
    ax_r.set_ylim(-span, span)
    ax_r.add_patch(plt.Circle((0, 0), p_star, fill=False, ls="--", color="#9ca3af"))
    ax_r.set_title("parent rest frame", fontsize=10)
    ax_r.set_xlabel("$p_z$ (GeV)")
    ax_r.set_ylabel("$p_x$ (GeV)")
    angle_text = ax_r.text(0.03, 0.95, "", transform=ax_r.transAxes, va="top", fontsize=9)

    zmax = max(l1[:, 0].max(), l2[:, 0].max(), p_par[0]) * 1.1
    zmin = min(l1[:, 0].min(), l2[:, 0].min(), 0.0) * 1.1 - 0.05 * zmax
    xmax = max(np.abs(l1[:, 1]).max(), np.abs(l2[:, 1]).max()) * 1.4
    ax_l.set_xlim(zmin, zmax)
    ax_l.set_ylim(-xmax, xmax)
    ax_l.set_title(f"lab frame: {title}", fontsize=10)
    ax_l.set_xlabel("$p_z$ (GeV)")
    for ax in (ax_r, ax_l):
        ax.set_aspect("equal", adjustable="box")
    ax_l.annotate("", xy=p_par, xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": CP, "lw": 1, "ls": "--"})
    ax_l.text(p_par[0], 0.06 * xmax, "parent", color=CP, fontsize=8, ha="right")
    (trace1,) = ax_l.plot([], [], color=C1, lw=1, alpha=0.5)
    (trace2,) = ax_l.plot([], [], color=C2, lw=1, alpha=0.5)
    readout = ax_l.text(0.02, 0.04, "", transform=ax_l.transAxes, fontsize=8, family="monospace")

    arrows: list = []

    def arrow(ax, tip, tail, colour, label):
        a = ax.annotate("", xy=tip, xytext=tail, arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2})
        t = ax.text(*(tip + 0.04 * span), label, color=colour, fontsize=9)
        arrows.extend([a, t])

    def update(step: int):
        for artist in arrows:
            artist.remove()
        arrows.clear()
        arrow(ax_r, r1[step], (0, 0), C1, name1)
        arrow(ax_r, r2[step], (0, 0), C2, name2)
        arrow(ax_l, l1[step], (0, 0), C1, name1)
        arrow(ax_l, l2[step], (0, 0), C2, name2)
        arrow(ax_l, l1[step] + l2[step], l1[step], C2, "")  # p2 drawn from the tip of p1: sum = parent
        trace1.set_data(l1[: step + 1, 0], l1[: step + 1, 1])
        trace2.set_data(l2[: step + 1, 0], l2[: step + 1, 1])
        angle_text.set_text(f"θ* = {deg[step]:.0f}°")
        readout.set_text(f"m_inv = √((p1+p2)·(p1+p2)) = {mass[step]:.4f} GeV")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(deg), 90), path, fps=15)
