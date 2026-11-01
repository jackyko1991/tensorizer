"""Animate the temperature field with heat-flux arrows, next to the plate's total heat content."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(
    times: torch.Tensor,
    frames: torch.Tensor,
    flux: torch.Tensor,
    heat: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    t_range: tuple[float, float],
    path: Path,
) -> Path:
    T = frames.cpu().numpy()  # (F, H, W)
    q = flux.cpu().numpy()  # (F, 2, H, W)
    t = times.cpu().numpy()
    heat = (heat / heat[0]).cpu().numpy()  # (F,) relative to the start
    x_cm, y_cm = 100 * X.cpu().numpy(), 100 * Y.cpu().numpy()
    side = 100 * (X[0, -1] + X[0, 0]).item()  # plate side in cm
    s = max(1, T.shape[-1] // 8)  # draw one arrow every s cells
    arrow_cm = 0.8 * s * (x_cm[0, 1] - x_cm[0, 0])  # length of the longest arrow

    fig, (ax, ax_q) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1.25, 1]})
    image = ax.imshow(T[0], origin="lower", extent=(0, side, 0, side), cmap="inferno", vmin=t_range[0], vmax=t_range[1])
    fig.colorbar(image, ax=ax, label="T (°C)", fraction=0.046, pad=0.03)
    arrows = ax.quiver(
        x_cm[::s, ::s], y_cm[::s, ::s], 0 * x_cm[::s, ::s], 0 * y_cm[::s, ::s],
        color="#93c5fd", angles="xy", scale_units="xy", scale=1 / arrow_cm,
    )  # fmt: skip
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.grid(False)
    title = ax.set_title("")

    ax_q.set_xlim(0, t[-1])
    ax_q.set_ylim(0, 1.1)
    ax_q.set_xlabel("t (s)")
    ax_q.set_ylabel("Q(t) / Q(0)")
    ax_q.set_title("heat content ∫ρc(T − T_edge) dA")
    (curve,) = ax_q.plot([], [], lw=2)

    def update(step: int):
        image.set_data(T[step])
        q_max = max(1e-12, float((q[step, 0] ** 2 + q[step, 1] ** 2).max() ** 0.5))
        arrows.set_UVC(q[step, 0, ::s, ::s] / q_max, q[step, 1, ::s, ::s] / q_max)  # rescaled each frame
        curve.set_data(t[: step + 1], heat[: step + 1])
        title.set_text(f"t = {t[step]:.1f} s,  max T = {T[step].max():.0f} °C")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t)), path)
