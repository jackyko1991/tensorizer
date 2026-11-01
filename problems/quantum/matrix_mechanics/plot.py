"""Animate the Heisenberg-picture matrix X(t), the commutator, and ⟨X⟩, ⟨P⟩ of the coherent state."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    times: torch.Tensor,
    X_t: torch.Tensor,
    C: torch.Tensor,
    heis: dict,
    schr: dict,
    alpha: float,
    lam: float,
    path: Path,
    block: int = 10,
) -> Path:
    t = times.cpu().numpy()
    Xb = X_t[:, :block, :block].real.cpu().numpy()  # (T, block, block) top-left corner of Re X(t)
    Ci = C.imag.cpu().numpy()  # (N, N): [X, P] / i
    xh, ph = heis["X"].cpu().numpy(), heis["P"].cpu().numpy()  # (T,)
    xs = schr["X"].cpu().numpy()  # (T,) the same numbers from the Schrödinger picture
    N = Ci.shape[0]
    r = math.sqrt(2) * alpha  # radius of the classical circle

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6))
    (ax_x, ax_c), (ax_ph, ax_t) = axes

    # Re X(t)_{mn}: the entries next to the diagonal rotate like e^{±it}
    vmax = np.abs(Xb).max()
    levels = np.linspace(-vmax, vmax, 12)
    img = ax_x.imshow(Xb[0], cmap="RdBu_r", norm=BoundaryNorm(levels, 256))
    ax_x.set_title(f"Re X(t) = Re U†XU   (first {block} states)", fontsize=9)
    ax_x.set_xlabel("n")
    ax_x.set_ylabel("m")
    ax_x.grid(False)
    ax_x.set_xticks(range(0, block, 3))
    ax_x.set_yticks(range(0, block, 3))
    cbar_x = fig.colorbar(img, ax=ax_x, shrink=0.85, ticks=[-vmax, 0, vmax], format="%.2f")
    cbar_x.ax.tick_params(labelsize=7)

    # [X, P]/i: the identity, except the bottom-right corner 1 − N
    c_levels = [1 - N - 0.5, -0.5, 0.5, 1.5]
    img_c = ax_c.imshow(Ci, cmap="PuOr", norm=BoundaryNorm(c_levels, 256))
    ax_c.set_title(f"[X, P] / i   (N = {N})", fontsize=9)
    ax_c.annotate(
        f"{1 - N}",
        (N - 1, N - 1),
        xytext=(0.45 * N, 0.85 * N),
        fontsize=8,
        arrowprops={"arrowstyle": "->"},
    )
    ax_c.set_xlabel("n")
    ax_c.grid(False)
    cbar = fig.colorbar(img_c, ax=ax_c, shrink=0.85, ticks=[1 - N, 0, 1])
    cbar.ax.tick_params(labelsize=7)

    # phase space: ⟨X⟩, ⟨P⟩ of the coherent state against the classical circle
    angle = np.linspace(0, 2 * np.pi, 200)
    ax_ph.plot(r * np.cos(angle), -r * np.sin(angle), "--", color="#9ca3af", lw=1, label="classical circle")
    (trail,) = ax_ph.plot([], [], color=ACCENT[0], lw=1.5, label="⟨X⟩, ⟨P⟩")
    (dot,) = ax_ph.plot([], [], "o", color=ACCENT[0], ms=7)
    lim = 1.2 * max(r, np.abs(xh).max(), np.abs(ph).max(), 0.5)
    ax_ph.set_xlim(-lim, lim)
    ax_ph.set_ylim(-lim, lim)
    ax_ph.set_aspect("equal")
    ax_ph.set_xlabel("⟨X⟩")
    ax_ph.set_ylabel("⟨P⟩")
    ax_ph.set_title(f"⟨X⟩, ⟨P⟩ of |α = {alpha:g}⟩,  λ = {lam:g}   (dashed: classical)", fontsize=9)

    # ⟨X⟩(t) from both pictures
    ax_t.plot(t, r * np.cos(t), "--", color="#9ca3af", lw=1, label="√2 Re(α e^(−it))")
    (line_h,) = ax_t.plot([], [], color=ACCENT[0], lw=2, label="Heisenberg: c† X(t) c")
    (dots_s,) = ax_t.plot([], [], "o", color=ACCENT[1], ms=3, label="Schrödinger: c(t)† X c(t)")
    ax_t.set_xlim(0, t[-1])
    ax_t.set_ylim(-1.5 * lim, 1.5 * lim)
    ax_t.set_xlabel("t")
    ax_t.set_title("⟨X⟩(t)", fontsize=9)
    ax_t.legend(loc="upper right", fontsize=7, ncol=1)
    clock = ax_t.text(0.02, 0.05, "", transform=ax_t.transAxes, family="monospace", fontsize=8)

    def update(step: int):
        img.set_data(Xb[step])
        trail.set_data(xh[: step + 1], ph[: step + 1])
        dot.set_data([xh[step]], [ph[step]])
        line_h.set_data(t[: step + 1], xh[: step + 1])
        dots_s.set_data(t[: step + 1 : 4], xs[: step + 1 : 4])
        clock.set_text(f"t = {t[step]:5.2f}   |Heis − Schr| = {abs(xh[step] - xs[step]):.0e}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 60), path)
