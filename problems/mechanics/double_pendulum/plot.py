"""Animate the two double pendulums on top of each other, with the growing gap between them beside."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .solve import positions


def animate(times: torch.Tensor, traj: torch.Tensor, L1: float, L2: float, path: Path, trail: int = 25) -> Path:
    """trail = number of animation frames that the lower bob's tail remembers."""
    t = times.cpu().numpy()  # (T,)
    q = traj[:, :, 0].cpu()  # (T, B, 2)
    bob1, bob2 = (p.numpy() for p in positions(q, L1, L2))  # (T, B, 2) each
    gap = (q[:, 0] - q[:, 1]).norm(dim=-1).clamp_min(1e-12).log10().numpy()  # (T,) log10 distance of the two runs
    frames = frame_indices(len(t), 100)
    reach = L1 + L2

    fig, (ax_swing, ax_trace) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1, 1.3]})

    # --- left: the pendulums ---
    ax_swing.set_xlim(-1.1 * reach, 1.1 * reach)
    ax_swing.set_ylim(-1.1 * reach, 1.1 * reach)
    ax_swing.set_aspect("equal", adjustable="box")
    ax_swing.set_xticks([])
    ax_swing.set_yticks([])
    ax_swing.plot([0], [0], "k^", ms=6)
    arms = [ax_swing.plot([], [], "o-", lw=2, ms=6, color=ACCENT[b], alpha=0.85)[0] for b in range(2)]
    tails = [ax_swing.plot([], [], lw=1, color=ACCENT[b], alpha=0.5)[0] for b in range(2)]
    clock = ax_swing.text(0.03, 0.97, "", transform=ax_swing.transAxes, va="top", fontsize=9)

    # --- right: how far apart the two runs are, in powers of ten (a straight rise = exponential growth) ---
    ax_trace.set_xlim(t[0], t[-1])
    ax_trace.set_ylim(max(gap.min(), gap[0] - 2) - 0.5, gap.max() + 0.5)
    ax_trace.set_xlabel("t (s)")
    ax_trace.set_ylabel("log₁₀ |q_A − q_B|  (rad)")
    ax_trace.set_title("two starts differing by a tiny δ in θ₂", fontsize=9)
    trace = ax_trace.plot([], [], lw=1.2, color="#111827")[0]

    def update(step: int):
        k = frames.index(step)
        start = frames[max(0, k - trail)]  # time step where the tail begins
        for b in range(2):
            arms[b].set_data([0, bob1[step, b, 0], bob2[step, b, 0]], [0, bob1[step, b, 1], bob2[step, b, 1]])
            tails[b].set_data(bob2[start : step + 1, b, 0], bob2[start : step + 1, b, 1])
        trace.set_data(t[: step + 1], gap[: step + 1])
        clock.set_text(f"t = {t[step]:.2f} s")

    fig.tight_layout()
    return save_gif(fig, update, frames, path)
