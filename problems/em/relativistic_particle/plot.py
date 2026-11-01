"""Animate relativistic (solid) against Newtonian (dashed) motion, with γ or the number of turns alongside."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import frame_indices, save_gif


def animate(lab: dict, newton: np.ndarray, config: str, labels: list[str], strength: float, path: Path) -> Path:
    t, x, gamma, turns = lab["t"], lab["x"], lab["gamma"], lab["turns"]  # (F,), (F, P, 3), (F, P), (F, P)
    n_particles = x.shape[1]
    colours = [f"C{p}" for p in range(n_particles)]
    fig, (ax, ax_right) = plt.subplots(1, 2, figsize=(7, 4))

    if config == "cyclotron":
        # left: orbits in the x–y plane; right: turns completed, Newton predicts ω = qB/m for everyone
        for p in range(n_particles):
            ax.plot(newton[:, p, 0], newton[:, p, 1], "--", color=colours[p], lw=0.8, alpha=0.6)
        ax.set_aspect("equal")
        lo, hi = min(x[..., :2].min(), newton[..., :2].min()), max(x[..., :2].max(), newton[..., :2].max())
        ax.set_xlim(lo - 0.3, hi + 0.3)
        ax.set_ylim(lo - 0.3, hi + 0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("orbits in B_z (dashed: Newton)")
        horizontal, vertical, right = x[..., 0], x[..., 1], turns
        ax_right.plot(t, strength * t / (2 * np.pi), "k--", lw=1, label="Newton: ω = qB/m")
        ax_right.set_ylim(0, strength * t[-1] / (2 * np.pi) * 1.05)
        ax_right.set_ylabel("turns completed")
        ax_right.set_title("relativistic: ω = qB/(γm)")
    else:
        # left: x(t) can never cross the light line x = t; right: γ grows without limit
        ax.plot(t, t, "k:", lw=1, label="light: x = t")
        for p in range(n_particles):
            ax.plot(t, newton[:, p, 0], "--", color=colours[p], lw=0.8, alpha=0.6)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0, t[-1] * 1.05)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title("pushed by E_x (dashed: Newton ½at²)")
        horizontal, vertical, right = np.broadcast_to(t[:, None], gamma.shape), x[..., 0], gamma
        ax_right.set_ylim(0, gamma.max() * 1.05)
        ax_right.set_ylabel("γ = U⁰")
        ax_right.set_title("Lorentz factor γ(t)")

    trails = [ax.plot([], [], color=c, lw=1.6, label=lab)[0] for c, lab in zip(colours, labels)]
    dots = [ax.plot([], [], "o", ms=5, color=c)[0] for c in colours]
    curves = [ax_right.plot([], [], color=c, lw=1.6)[0] for c in colours]
    ax.legend(loc="upper left", fontsize=7)
    ax_right.set_xlim(0, t[-1])
    ax_right.set_xlabel("lab time t")
    if config == "cyclotron":
        ax_right.legend(loc="upper left", fontsize=8)

    def update(step: int):
        for p in range(n_particles):
            trails[p].set_data(horizontal[: step + 1, p], vertical[: step + 1, p])
            dots[p].set_data([horizontal[step, p]], [vertical[step, p]])
            curves[p].set_data(t[: step + 1], right[: step + 1, p])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 80), path)
