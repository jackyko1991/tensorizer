"""Animate the ensemble: faint noisy trajectories around the noise-free curve, the histogram of x, and ⟨x²⟩, ⟨v²⟩."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

N_DRAWN = 24  # trajectories drawn faintly (all B particles enter the statistics)
N_BINS = 31


def animate(
    times: torch.Tensor,
    x: torch.Tensor,
    x_det: torch.Tensor,
    x2: torch.Tensor,
    v2: torch.Tensor,
    x2_eq: float,
    v2_eq: float,
    path: Path,
) -> Path:
    t = times.cpu().numpy()
    xs = x.cpu().numpy()  # (S, B)
    xd, m_x2, m_v2 = x_det.cpu().numpy(), x2.cpu().numpy(), v2.cpu().numpy()
    sigma = np.sqrt(x2_eq)
    lim = max(4 * sigma, 1.1 * np.abs(xd).max(), 1e-3)
    edges = np.linspace(-lim, lim, N_BINS + 1)
    width = edges[1] - edges[0]
    counts = np.stack([np.histogram(row, bins=edges)[0] for row in xs]) / (xs.shape[1] * width)  # (S, bins) density

    fig = plt.figure(figsize=(8.2, 4.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
    ax_t = fig.add_subplot(grid[:, 0])
    ax_h = fig.add_subplot(grid[0, 1])
    ax_m = fig.add_subplot(grid[1, 1])

    # --- trajectories x_b(t) ---
    ax_t.set_xlim(0, t[-1])
    ax_t.set_ylim(-lim, lim)
    ax_t.set_xlabel("time t")
    ax_t.set_ylabel("position x")
    ax_t.set_title("Langevin trajectories (noise) vs noise-free")
    if sigma > 0:
        ax_t.axhspan(-sigma, sigma, color="#93c5fd", alpha=0.25, lw=0)
    faint = [ax_t.plot([], [], lw=0.6, color="#2563eb", alpha=0.25)[0] for _ in range(min(N_DRAWN, xs.shape[1]))]
    (bold,) = ax_t.plot([], [], lw=2.4, color="#dc2626", label="noise-free (= mean)")
    (mean,) = ax_t.plot([], [], lw=1.2, color="#111827", ls="--", label=r"ensemble mean $\langle x\rangle$")
    ax_t.legend(loc="upper right", fontsize=8)
    clock = ax_t.text(0.02, 0.04, "", transform=ax_t.transAxes, fontsize=9)

    # --- histogram of x across the ensemble ---
    centers = 0.5 * (edges[1:] + edges[:-1])
    bars = ax_h.bar(centers, counts[0], width=width * 0.9, color="#93c5fd", edgecolor="#2563eb", lw=0.5)
    if sigma > 0:
        fine = np.linspace(-lim, lim, 200)
        ax_h.plot(fine, np.exp(-(fine**2) / (2 * x2_eq)) / np.sqrt(2 * np.pi * x2_eq), color="#dc2626", lw=1.5)
        ax_h.set_ylim(0, 1.6 / np.sqrt(2 * np.pi * x2_eq))
    else:
        ax_h.set_ylim(0, counts.max() * 1.1)
    ax_h.set_xlim(-lim, lim)
    ax_h.set_title(r"histogram of $x_b$ vs $N(0,\,k_BT/k)$", fontsize=10)

    # --- ⟨x²⟩(t) and ⟨v²⟩(t) approaching equipartition ---
    top = 1.15 * max(m_x2.max(), m_v2.max(), 1e-6)
    ax_m.set_xlim(0, t[-1])
    ax_m.set_ylim(0, top)
    ax_m.set_xlabel("time t", labelpad=1)
    ax_m.axhline(x2_eq, color="#2563eb", ls="--", lw=1)
    ax_m.axhline(v2_eq, color="#ea580c", ls=":", lw=1.2)
    (l_x2,) = ax_m.plot([], [], color="#2563eb", lw=1.6, label=r"$\langle x^2\rangle \to k_BT/k$")
    (l_v2,) = ax_m.plot([], [], color="#ea580c", lw=1.6, label=r"$\langle v^2\rangle \to k_BT/m$")
    ax_m.legend(loc="upper right", fontsize=7.5)
    mean_x = xs.mean(axis=1)

    def update(step: int):
        for b, line in enumerate(faint):
            line.set_data(t[: step + 1], xs[: step + 1, b])
        bold.set_data(t[: step + 1], xd[: step + 1])
        mean.set_data(t[: step + 1], mean_x[: step + 1])
        for bar, h in zip(bars, counts[step]):
            bar.set_height(h)
        l_x2.set_data(t[: step + 1], m_x2[: step + 1])
        l_v2.set_data(t[: step + 1], m_v2[: step + 1])
        clock.set_text(f"t = {t[step]:.1f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 72), path)
