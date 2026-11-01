"""Animate |ψ|² and Re ψ of the drifting, spreading packet, with the predicted ⟨x⟩ ± σ(t) marked."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif


def animate(
    x: torch.Tensor,
    times: torch.Tensor,
    psi: torch.Tensor,
    mean: torch.Tensor,
    width: torch.Tensor,
    path: Path,
) -> Path:
    xs = x.cpu().numpy()
    psi = psi.cpu()
    prob = (psi.abs() ** 2).numpy()  # (T, N)
    re = psi.real.numpy()  # (T, N)
    t = times.cpu().numpy()
    xm, sig = mean.cpu().numpy(), width.cpu().numpy()  # (T,), (T,)
    lo = min(xm.min() - 3 * sig.max(), xm[0] - 10)
    hi = max(xm.max() + 3 * sig.max(), xm[0] + 10)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(max(lo, xs[0]), min(hi, xs[-1]))
    amp = float(abs(re).max())
    ax.set_ylim(-1.1 * amp, 1.25 * amp)
    ax.set_xlabel("x")
    ax.set_ylabel("Re ψ,  |ψ|²")

    (line_re,) = ax.plot([], [], color=ACCENT[0], lw=0.9, alpha=0.8, label="Re ψ")
    (line_p,) = ax.plot([], [], color="#111827", lw=2.2, label="|ψ|²")
    fill = [ax.fill_between(xs, 0 * xs, color="#111827", alpha=0.1)]
    centre = ax.axvline(0, color=ACCENT[1], lw=1.2, label="x0 + k0 t")
    edges = [ax.axvline(0, color=ACCENT[1], lw=1, ls="--") for _ in range(2)]
    edges[0].set_label("± σ(t)")
    clock = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", family="monospace")
    ax.legend(loc="upper right", fontsize=8)

    def update(step: int):
        line_re.set_data(xs, re[step])
        line_p.set_data(xs, prob[step])
        fill[0].remove()
        fill[0] = ax.fill_between(xs, prob[step], color="#111827", alpha=0.1)
        centre.set_xdata([xm[step], xm[step]])
        edges[0].set_xdata([xm[step] - sig[step]] * 2)
        edges[1].set_xdata([xm[step] + sig[step]] * 2)
        clock.set_text(f"t = {t[step]:5.1f}\nσ = {sig[step]:5.2f}")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path)
