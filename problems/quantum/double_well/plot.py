"""Animate |ψ|² tunnelling between the wells (left) and the probability P_L(t) in the left well (right)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    x: torch.Tensor,
    times: torch.Tensor,
    psi: torch.Tensor,
    V: torch.Tensor,
    phi: torch.Tensor,
    energies: torch.Tensor,
    p_left: torch.Tensor,
    prediction: torch.Tensor,
    path: Path,
) -> Path:
    xs, Vs = x.cpu().numpy(), V.cpu().numpy()
    prob = (psi.abs() ** 2).cpu().numpy()  # (T, N)
    t = times.cpu().numpy()
    pl, pred = p_left.cpu().numpy(), prediction.cpu().numpy()
    E = energies[:2].cpu().numpy()
    doublet = phi[:, :2].T.cpu().numpy()  # (2, N)  φ_0 (even) and φ_1 (odd)
    V0 = float(V[len(V) // 2])  # barrier height at x ≈ 0

    fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    view = xs[Vs < 2.2 * V0]  # region where the potential is below twice the barrier
    ax.set_xlim(view.min(), view.max())
    ax.set_ylim(0, 2.0 * V0)
    ax.set_xlabel("x")
    ax.set_ylabel("V(x),  E_n")
    ax.plot(xs, Vs, color="#6b7280", lw=2)
    scale = 0.12 * V0 / abs(doublet).max()
    for En, state in zip(E, doublet):  # faint doublet φ_0, φ_1 drawn at their energies
        ax.plot(xs, En + scale * state, color="#9ca3af", lw=0.8, ls="--")

    ax_psi = ax.twinx()
    ax_psi.grid(False)
    ax_psi.set_ylim(0, 2.2 * prob.max())
    ax_psi.set_yticks([])
    (line,) = ax_psi.plot([], [], color=ACCENT[0], lw=2)
    fill = [ax_psi.fill_between(xs, 0 * xs, color=ACCENT[0], alpha=0.25)]
    clock = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", family="monospace")
    ax.set_title("|ψ|² (blue),  doublet φ0, φ1 (dashed)", fontsize=10)

    ax_p.set_xlim(t[0], t[-1])
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.set_xlabel("t")
    ax_p.set_ylabel("probability in left well")
    ax_p.plot(t, pred, color="#9ca3af", ls="--", lw=1, label="cos²(ΔE t/2)")
    (curve,) = ax_p.plot([], [], color=ACCENT[1], lw=2, label="P_L(t)")
    (dot,) = ax_p.plot([], [], "o", color=ACCENT[1])
    ax_p.legend(loc="lower left", fontsize=8)

    def update(step: int):
        line.set_data(xs, prob[step])
        fill[0].remove()
        fill[0] = ax_psi.fill_between(xs, prob[step], color=ACCENT[0], alpha=0.25)
        curve.set_data(t[: step + 1], pl[: step + 1])
        dot.set_data([t[step]], [pl[step]])
        clock.set_text(f"t = {t[step]:6.1f}\nP_L = {pl[step]:.3f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
