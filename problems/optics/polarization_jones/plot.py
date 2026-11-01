"""Animate the analyzer rotating: field ellipse and transmitted field (left), intensity vs angle (right)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(angles: torch.Tensor, res: dict, plate: str, path: Path) -> Path:
    theta = angles.numpy()  # (A,)
    ell_in, ell_plate = res["ellipse_in"].numpy(), res["ellipse_plate"].numpy()  # (T, 2) each
    i_out, i_malus = res["intensity"].numpy(), res["malus"].numpy()  # (A,) each
    n_phase = len(ell_plate)

    fig, (ax, ax_i) = plt.subplots(1, 2, figsize=(7, 4))
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_xlabel("Ex")
    ax.set_ylabel("Ey")
    ax.plot(ell_in[:, 0], ell_in[:, 1], "--", color="#9ca3af", lw=1.5, label="input")
    ax.plot(ell_plate[:, 0], ell_plate[:, 1], color="#2563eb", lw=1.5, label=f"after {plate} plate")
    (axis_line,) = ax.plot([], [], color="#374151", lw=0.8, ls="-.")
    (passed,) = ax.plot([], [], color="#dc2626", lw=3, label="after analyzer")
    (tip,) = ax.plot([], [], "o", color="#2563eb", ms=6)
    (proj,) = ax.plot([], [], ":", color="#dc2626", lw=1)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("field ellipse")

    deg = np.degrees(theta)
    ax_i.plot(deg, i_malus, "--", color="#9ca3af", label="no plate: cos²θ (Malus)")
    ax_i.plot(deg, i_out, color="#2563eb", label=f"with {plate} plate")
    (dot,) = ax_i.plot([], [], "o", color="#dc2626")
    ax_i.set_xlim(0, 180)
    ax_i.set_ylim(0, 1.15)
    ax_i.set_xticks([0, 45, 90, 135, 180])
    ax_i.set_xlabel("analyzer angle θ (°)")
    ax_i.set_ylabel("transmitted I / I₀")
    ax_i.legend(loc="upper right", fontsize=7)
    title = ax_i.set_title(" ")

    def update(a: int):
        u = np.array([np.cos(theta[a]), np.sin(theta[a])])  # analyzer transmission axis
        axis_line.set_data([-1.3 * u[0], 1.3 * u[0]], [-1.3 * u[1], 1.3 * u[1]])
        amp = np.sqrt(i_out[a])  # |E_out|: the passed field oscillates along u with this amplitude
        passed.set_data([-amp * u[0], amp * u[0]], [-amp * u[1], amp * u[1]])
        e_now = ell_plate[(7 * a) % n_phase]  # field at the current instant: it runs around the ellipse
        e_passed = (e_now @ u) * u  # the analyzer keeps the component along its axis
        tip.set_data([e_now[0]], [e_now[1]])
        proj.set_data([e_now[0], e_passed[0]], [e_now[1], e_passed[1]])
        dot.set_data([deg[a]], [i_out[a]])
        title.set_text(f"θ = {deg[a]:.0f}°,  I = {i_out[a]:.2f} I₀")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(theta), 90), path, fps=12)
