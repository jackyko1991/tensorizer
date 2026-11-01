"""Animate the sweep: F'^{μν} as an annotated heatmap next to bars for E' and B'."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

LABELS = ["t", "x", "y", "z"]


def animate(sweep: dict, path: Path) -> Path:
    beta = sweep["beta"].numpy()  # (S,)
    F = sweep["F"].numpy()  # (S, 4, 4)
    EB = np.concatenate([sweep["E"].numpy(), sweep["B"].numpy()], axis=1)  # (S, 6): E'x E'y E'z B'x B'y B'z
    FF, FFdual = sweep["FF"].numpy(), sweep["FFdual"].numpy()  # (S,), (S,)
    limit = max(np.abs(F).max(), 1e-6)

    fig, (ax_f, ax_v) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1, 1.1]})
    header = fig.suptitle("")

    # left: the 4x4 tensor, one annotated cell per component
    image = ax_f.imshow(F[0], cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax_f.set_xticks(range(4), [f"ν={s}" for s in LABELS])
    ax_f.set_yticks(range(4), [f"μ={s}" for s in LABELS])
    ax_f.grid(False)
    ax_f.set_title(r"$F'^{\mu\nu}$")
    cells = [[ax_f.text(n, m, "", ha="center", va="center", fontsize=9) for n in range(4)] for m in range(4)]

    # right: the six field components read back out of F'
    names = ["E'x", "E'y", "E'z", "B'x", "B'y", "B'z"]
    bars = ax_v.bar(names, EB[0], color=[ACCENT[0]] * 3 + [ACCENT[1]] * 3)
    top = max(np.abs(EB).max() * 1.15, 1e-6)
    ax_v.set_ylim(-top, top)
    ax_v.axhline(0, color="#6b7280", lw=0.8)
    ax_v.grid(axis="x", visible=False)
    ax_v.set_axisbelow(True)
    ax_v.set_title("fields seen by the moving observer")

    def update(step: int):
        image.set_data(F[step])
        for m in range(4):
            for n in range(4):
                value = F[step, m, n]
                cells[m][n].set_text(f"{value:+.2f}" if m != n else "0")
                cells[m][n].set_color("white" if abs(value) > 0.6 * limit else "black")
        for bar, height in zip(bars, EB[step]):
            bar.set_height(height)
        gamma = 1 / np.sqrt(1 - beta[step] ** 2)
        header.set_text(f"boost along x:  β = {beta[step]:.3f},  γ = {gamma:.2f}")
        # the two Lorentz invariants: the same for every β
        invariant = r"$F_{\mu\nu}F^{\mu\nu} = 2(B^2-E^2) = %+.3f$" % FF[step]
        pseudo = r"$\mathbf{E}\cdot\mathbf{B} = -\frac{1}{4}F_{\mu\nu}\tilde F^{\mu\nu} = %+.3f$" % (-FFdual[step] / 4)
        ax_v.set_xlabel(invariant + "\n" + pseudo)

    update(0)
    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(beta)), path)
