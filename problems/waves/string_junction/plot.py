"""Animate the string: the heavier/lighter half is shaded, dashed lines show the predicted R and 𝒯."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(times: torch.Tensor, x: torch.Tensor, traj: torch.Tensor, info: dict, path: Path) -> Path:
    t, xs, u = times.cpu().numpy(), x.cpu().numpy(), traj.cpu().numpy()  # (S,), (N,), (S, N)
    R, Tr = info["R"], info["T"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvspan(0, 1, color="#93c5fd", alpha=0.25, lw=0)
    ax.axvline(0, color="#374151", lw=1)
    ax.plot([-1, 0], [R, R], "--", color="#9ca3af", lw=1)
    ax.plot([0, 1], [Tr, Tr], "--", color="#9ca3af", lw=1)
    ax.text(-0.98, R, f" R = {R:+.3f}", va="bottom" if R >= 0 else "top", fontsize=9, color="#4b5563")
    ax.text(0.98, Tr, rf"$\mathcal{{T}}$ = {Tr:.3f} ", ha="right", va="bottom", fontsize=9, color="#4b5563")
    ax.text(-0.5, 1.3, f"μ₁ = 1,  Z₁ = {info['Z1']:.2f}", ha="center", fontsize=9)
    ax.text(0.5, 1.3, f"μ₂ = {info['mu2']:.2g},  Z₂ = {info['Z2']:.2f}", ha="center", fontsize=9)
    line = ax.plot(xs, u[0], color="#2563eb", lw=2)[0]
    ax.plot([-1, 1], [0, 0], "o", color="#374151", ms=5)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.1, 1.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("u")
    title = ax.set_title("string displacement   t = 0.00 s")

    def update(step: int):
        line.set_ydata(u[step])
        title.set_text(f"string displacement   t = {t[step]:.2f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
