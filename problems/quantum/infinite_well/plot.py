"""Animate |ψ|², Re ψ and Im ψ inside the box, with the lowest eigenstates drawn faintly behind."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def with_walls(f: torch.Tensor) -> torch.Tensor:
    """Append the boundary values ψ(0) = ψ(L) = 0 along the last axis: (..., N) → (..., N+2)."""
    zero = f.new_zeros(*f.shape[:-1], 1)
    return torch.cat([zero, f, zero], dim=-1)


def animate(
    x: torch.Tensor, times: torch.Tensor, psi: torch.Tensor, phi: torch.Tensor, n_show: int, L: float, path: Path
) -> Path:
    xs = torch.cat([x.new_zeros(1), x, x.new_full((1,), L)]).cpu().numpy()  # (N+2,) grid incl. walls
    psi = with_walls(psi).cpu()  # (T, N+2)
    prob = (psi.abs() ** 2).numpy()
    re, im = psi.real.numpy(), psi.imag.numpy()
    states = with_walls(phi[:, :n_show].T).cpu().numpy()  # (n_show, N+2)
    t = times.cpu().numpy()
    amp = float(psi.abs().max())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, L)
    ax.set_ylim(-1.1 * amp, 1.1 * max(prob.max(), amp))
    ax.set_xlabel("x")
    ax.set_ylabel("ψ,  |ψ|²")
    for wall in (0, L):
        ax.axvline(wall, color="#374151", lw=3)
    for state in states:  # faint stationary states φ_n(x)
        ax.plot(xs, state, color="#9ca3af", lw=0.8, ls="--")

    (line_re,) = ax.plot([], [], color=ACCENT[0], lw=1.2, label="Re ψ")
    (line_im,) = ax.plot([], [], color=ACCENT[1], lw=1.2, label="Im ψ")
    (line_p,) = ax.plot([], [], color="#111827", lw=2.2, label="|ψ|²")
    fill = [ax.fill_between(xs, 0 * xs, color="#111827", alpha=0.08)]
    clock = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right", fontsize=8)

    def update(step: int):
        line_re.set_data(xs, re[step])
        line_im.set_data(xs, im[step])
        line_p.set_data(xs, prob[step])
        fill[0].remove()
        fill[0] = ax.fill_between(xs, prob[step], color="#111827", alpha=0.08)
        clock.set_text(f"t = {t[step]:.3f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path)
