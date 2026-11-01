"""Animate |ψ|² hitting the barrier, with live transmitted / reflected probabilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    x: torch.Tensor,
    times: torch.Tensor,
    psi: torch.Tensor,
    V: torch.Tensor,
    E: float,
    p_trans: torch.Tensor,
    path: Path,
    x_view: float = 60.0,
) -> Path:
    xs = x.cpu().numpy()
    prob = (psi.abs() ** 2).cpu().numpy()  # (T, N)
    pt = p_trans.cpu().numpy()
    t = times.cpu().numpy()
    top = 1.15 * prob.max()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(-x_view, x_view)
    ax.set_ylim(0, top)
    ax.set_xlabel("x")
    ax.set_ylabel("|ψ|²")

    # Barrier and packet energy on a second axis in energy units.
    ax_v = ax.twinx()
    ax_v.grid(False)
    v_top = 1.6 * max(float(V.max()), E, 1e-9)
    ax_v.set_ylim(0, v_top)
    ax_v.set_ylabel("V(x),  E", color="#6b7280")
    ax_v.fill_between(xs, V.cpu().numpy(), color="#6b7280", alpha=0.35, step="mid", label="barrier V(x)")
    ax_v.axhline(E, color=ACCENT[4], ls="--", lw=1, label="E = k0²/2")
    ax_v.legend(loc="upper right", fontsize=8)

    (line,) = ax.plot([], [], color=ACCENT[0], lw=2)
    fill = [ax.fill_between(xs, 0 * xs, color=ACCENT[0], alpha=0.2)]
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", family="monospace")

    def update(step: int):
        line.set_data(xs, prob[step])
        fill[0].remove()
        fill[0] = ax.fill_between(xs, prob[step], color=ACCENT[0], alpha=0.2)
        label.set_text(f"t = {t[step]:5.1f}\nP(right) = {pt[step]:.3f}\nP(left)  = {1 - pt[step]:.3f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path)
