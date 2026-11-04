"""Dots on the unit circle clustering as K ramps up, beside r(K) and a chimera snapshot on a ring."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from .equation import ott_antonsen_r

SHOWN = 400  # oscillators drawn on the circle


def animate(
    omega: torch.Tensor,
    k_t: torch.Tensor,
    thetas: torch.Tensor,
    r_t: torch.Tensor,
    k: torch.Tensor,
    sweeps: dict,
    width: float,
    dist: str,
    ring: tuple[torch.Tensor, torch.Tensor],
    path: Path,
) -> Path:
    """omega (N,); k_t, r_t (F,) and thetas (F, N) along the ramp; k (B,) and sweeps {dist: (r (B,), K_c)}."""
    th = thetas[:, :SHOWN].numpy()
    psi = torch.exp(1j * thetas).mean(-1).angle().numpy()  # (F,) mean-field direction
    w = omega[:SHOWN].clamp(-3 * width, 3 * width).numpy()
    kk, rr = k_t.numpy(), r_t.numpy()

    fig = plt.figure(figsize=(11, 4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.1], wspace=0.32)
    ax_c = fig.add_subplot(grid[0])
    circle = np.linspace(0, 2 * np.pi, 200)
    ax_c.plot(np.cos(circle), np.sin(circle), color="#d1d5db", lw=1)
    dots = ax_c.scatter(np.cos(th[0]), np.sin(th[0]), c=w, cmap="coolwarm", s=9, vmin=-3 * width, vmax=3 * width)
    (arrow,) = ax_c.plot([0, 0], [0, 0], color="#111827", lw=2.5)
    ax_c.set_xlim(-1.15, 1.15)
    ax_c.set_ylim(-1.15, 1.15)
    ax_c.set_aspect("equal")
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.grid(False)
    title = ax_c.set_title("", fontsize=10)
    ax_c.text(0, -1.32, "colour: natural frequency ω (mean field → up)", ha="center", fontsize=7, color="#6b7280")

    ax_r = fig.add_subplot(grid[1])
    k_fine = torch.linspace(1e-3, float(k.max()), 300)
    ax_r.plot(k_fine, ott_antonsen_r(k_fine, width), color="#111827", lw=1.5, label="Ott–Antonsen √(1 − K_c/K)")
    r_l, kc_l = sweeps["lorentzian"]
    r_g, kc_g = sweeps["gaussian"]
    ax_r.plot(k, r_l, "o", color="#2563eb", ms=4, label="Lorentzian (batch of K)")
    ax_r.plot(k, r_g, "s", color="#16a34a", ms=3.5, mfc="none", label="Gaussian (batch of K)")
    ax_r.axvline(kc_l, color="#2563eb", ls="--", lw=1)
    ax_r.axvline(kc_g, color="#16a34a", ls=":", lw=1)
    (trace,) = ax_r.plot([], [], color="#dc2626", lw=1.3, label=f"slow ramp ({dist})")
    (dot,) = ax_r.plot([], [], "o", color="#dc2626", ms=6)
    ax_r.set_xlim(0, float(k.max()))
    ax_r.set_ylim(0, 1.02)
    ax_r.set_xlabel("coupling K")
    ax_r.set_ylabel("order parameter r")
    ax_r.set_title(f"phase transition at K_c = 2/(π g(0))  ({kc_l:.2f}, {kc_g:.2f})", fontsize=9)
    ax_r.legend(fontsize=7, frameon=False, loc="lower right")

    x, theta = ring
    ax_x = fig.add_subplot(grid[2])
    ax_x.plot(x.numpy(), theta.numpy(), ".", color="#9333ea", ms=3)
    ax_x.set_xlim(-np.pi, np.pi)
    ax_x.set_ylim(-np.pi, np.pi)
    ax_x.set_xlabel("position on the ring x")
    ax_x.set_ylabel("phase θ")
    ax_x.set_title("chimera: identical oscillators,\none locked domain beside a drifting one", fontsize=9)

    def update(f: int):
        a = th[f] - psi[f] + np.pi / 2  # rotate so that the mean field points up
        dots.set_offsets(np.c_[np.cos(a), np.sin(a)])
        arrow.set_data([0, 0], [0, rr[f]])
        title.set_text(f"K = {kk[f]:.2f},  r = {rr[f]:.2f}")
        trace.set_data(kk[: f + 1], rr[: f + 1])
        dot.set_data([kk[f]], [rr[f]])

    return save_gif(fig, update, frame_indices(len(kk), 60), path, fps=6)
