"""Left: dye lines across the pipe bend into the Hagen–Poiseuille parabola. Right: the velocity profile u_z(r, t)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def mirror(a: np.ndarray) -> np.ndarray:
    """Values on r = 0 … R → values across the whole diameter, r = -R … R (the flow is axisymmetric)."""
    return np.concatenate([a[..., :0:-1], a], axis=-1)


def animate(
    times: torch.Tensor, r: torch.Tensor, u: torch.Tensor, C: torch.Tensor, releases: torch.Tensor,
    u_steady: torch.Tensor, R: float, tau: float, path: Path,
) -> Path:  # fmt: skip
    t = times.cpu().numpy()
    mm = 1000  # plot in millimetres
    ys = mirror(r.cpu().numpy()) * mm
    ys[: len(ys) // 2] *= -1  # left half of the diameter is negative
    us, cs, us_s = (mirror(a.cpu().numpy()) * mm for a in (u, C, u_steady))
    k_release = np.searchsorted(t, releases.cpu().numpy())  # time step of each release

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    window = us_s.max() * 0.12 * tau  # how far the steady centreline flow moves in 0.12 R²/ν
    ax.set_xlim(-0.03 * window, window)
    ax.set_ylim(-R * mm * 1.15, R * mm * 1.15)
    ax.fill_between([-window, 2 * window], -R * mm, R * mm, color="#eff6ff")
    for yw in (-R * mm, R * mm):
        ax.axhline(yw, color="#374151", lw=3)
    ax.axvline(0, color="#9ca3af", lw=1, ls="--")
    ax.set_xlabel("z (mm)   dashed: bubble wire")
    ax.set_ylabel("r (mm)")
    ax.grid(False)
    lines = [ax.plot([], [], color="#2563eb", lw=1.5)[0] for _ in k_release]
    title = ax.set_title(" ")

    ax2.plot(us_s, ys, color="#111827", ls="--", lw=1, label="Hagen–Poiseuille")
    profile = ax2.plot(us[0], ys, color="#dc2626", lw=2, label="u_z(r, t)")[0]
    ax2.set_xlim(0, us_s.max() * 1.1)
    ax2.set_ylim(-R * mm * 1.15, R * mm * 1.15)
    ax2.set_xlabel("u_z (mm/s)")
    ax2.legend(loc="lower right", fontsize=7)

    def update(step: int):
        for line, k in zip(lines, k_release):
            if k <= step:
                line.set_data(cs[step] - cs[k], ys)  # z_k(r, t) = C(t, r) - C(t_k, r)
            else:
                line.set_data([], [])
        profile.set_xdata(us[step])
        title.set_text(f"t = {t[step]:.1f} s = {t[step] / tau:.3f} R²/ν")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 75), path, fps=12)
