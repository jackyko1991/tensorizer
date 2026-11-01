"""Left: alpha trajectories past the nucleus. Right: the angle histogram filling up against Rutherford's law."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(
    paths: torch.Tensor,
    first: int,
    last: int,
    theta: torch.Tensor,
    detect_step: torch.Tensor,
    theta_grid: torch.Tensor,
    expected: torch.Tensor,
    d: float,
    path: Path,
) -> Path:
    """paths (T, S, 2, 2); frames run from step `first` to `last`; theta (B,) final angles of the beam;
    expected (G,) Rutherford prediction of dN/dΩ on theta_grid (G,)."""
    pos = paths[:, :, 0, :].cpu().numpy()  # (T, S, 2) in fm
    deg = np.degrees(theta.cpu().numpy())
    detected = detect_step.cpu().numpy()
    view = 5 * d

    fig, (ax_xy, ax_h) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax_xy.set_xlim(-view, view)
    ax_xy.set_ylim(-0.8 * view, 0.8 * view)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x (fm)")
    ax_xy.set_ylabel("y (fm)")
    ax_xy.plot(0, 0, "o", color="#dc2626", ms=7, zorder=5)
    ax_xy.add_patch(plt.Circle((0, 0), d, fill=False, ls="--", color="#9ca3af"))
    ax_xy.text(0.7 * d, -1.0 * d, "d = k/E", fontsize=8, color="#6b7280")
    lines = [ax_xy.plot([], [], lw=1.2, color="#2563eb")[0] for _ in range(pos.shape[1])]
    dots = ax_xy.plot([], [], "o", color="#1e40af", ms=3)[0]

    bins = np.linspace(max(0.0, deg.min()), 180.0, 19)
    solid_angle = 2 * np.pi * (np.cos(np.radians(bins[:-1])) - np.cos(np.radians(bins[1:])))  # sr per bin
    centres = 0.5 * (bins[:-1] + bins[1:])
    ax_h.plot(np.degrees(theta_grid.cpu().numpy()), expected.cpu().numpy(), color="#9ca3af", lw=2, ls="--")
    ax_h.plot([], [], color="#9ca3af", lw=2, ls="--", label=r"Rutherford $\propto 1/\sin^4(\theta/2)$")
    (hist_line,) = ax_h.plot([], [], "o-", color="#2563eb", ms=4, lw=1, label="simulated alphas")
    ax_h.set_yscale("log")
    ax_h.set_xlim(0, 180)
    exp_np = expected.cpu().numpy()
    ax_h.set_ylim(exp_np.min() * 0.2, exp_np.max() * 3)
    ax_h.set_xlabel("scattering angle θ (°)")
    ax_h.set_ylabel("dN/dΩ (per sr)")
    ax_h.legend(loc="upper right", fontsize=8)
    counter = ax_h.text(0.03, 0.04, "", transform=ax_h.transAxes, fontsize=9)

    def update(step: int):
        for s, line in enumerate(lines):
            line.set_data(pos[: step + 1, s, 0], pos[: step + 1, s, 1])
        dots.set_data(pos[step, :, 0], pos[step, :, 1])
        seen = deg[detected <= step]
        counts, _ = np.histogram(seen, bins=bins)
        density = np.where(counts > 0, counts / solid_angle, np.nan)
        hist_line.set_data(centres, density)
        counter.set_text(f"{seen.size} / {deg.size} alphas detected")

    fig.tight_layout()
    frames = [first + i for i in frame_indices(last - first + 1, 60)]
    return save_gif(fig, update, frames, path)
