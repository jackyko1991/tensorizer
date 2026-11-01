"""Animate the populations N_i(t) (exact curves, RK4 dots) and the activities approaching equilibrium."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    t: torch.Tensor,
    n: torch.Tensor,
    t_rk4: torch.Tensor,
    n_rk4: torch.Tensor,
    act: torch.Tensor,
    names: list[str],
    path: Path,
) -> Path:
    t, n, act = t.cpu().numpy(), n.cpu().numpy(), act.cpu().numpy()  # (T,), (T, M), (T, M)
    t_rk4, n_rk4 = t_rk4.cpu().numpy(), n_rk4.cpu().numpy()
    n_members = n.shape[1]

    fig, (ax_n, ax_a) = plt.subplots(1, 2, figsize=(8.5, 4))
    lines, dots = [], []
    for i in range(n_members):
        lines.append(ax_n.plot([], [], color=ACCENT[i], lw=2, label=names[i])[0])
        dots.append(ax_n.plot([], [], "o", color=ACCENT[i], ms=3.5, mfc="white")[0])
    ax_n.set_xlim(0, t[-1])
    ax_n.set_ylim(0, n.max() * 1.05)
    ax_n.set_xlabel("t (h)")
    ax_n.set_ylabel("number of nuclei $N_i$")
    ax_n.set_title("lines: $e^{At}$   circles: RK4", fontsize=9)
    ax_n.legend(loc="center right", fontsize=8)

    act_lines = [
        ax_a.plot([], [], color=ACCENT[i], lw=2, label=f"{names[i]} activity")[0] for i in range(n_members - 1)
    ]
    ax_a.set_yscale("log")
    ax_a.set_xlim(0, t[-1])
    ax_a.set_ylim(max(act[:, 0].min() * 0.2, act.max() * 1e-4), act.max() * 2)
    ax_a.set_xlabel("t (h)")
    ax_a.set_ylabel(r"activity $\lambda_i N_i$ (1/h)")
    ax_a.legend(loc="lower left", fontsize=8)
    ratio_text = ax_a.text(0.97, 0.95, "", transform=ax_a.transAxes, ha="right", va="top", fontsize=9)

    every = max(1, len(t_rk4) // 25)  # show every few RK4 steps as circles

    def update(step: int):
        for i in range(n_members):
            lines[i].set_data(t[: step + 1], n[: step + 1, i])
            shown = t_rk4 <= t[step]
            dots[i].set_data(t_rk4[shown][::every], n_rk4[shown][::every, i])
        for i, line in enumerate(act_lines):
            line.set_data(t[: step + 1], act[: step + 1, i])
        if act[step, 0] > 0:
            ratio_text.set_text(f"t = {t[step]:.0f} h\n$a_2/a_1$ = {act[step, 1] / act[step, 0]:.3f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
