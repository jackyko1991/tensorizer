"""Animate the growing wavefronts, the moving source, the Mach cone (if any) and what each listener hears."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle

from problems._common.plotting import frame_indices, save_gif


def animate(result: dict, t_e: torch.Tensor, s: torch.Tensor, obs: torch.Tensor, v_s: float, c: float, path: Path):
    times = result["times"].cpu().numpy()  # (T,)
    radii = result["radii"].cpu().numpy()  # (T, E)
    arrivals = result["arrivals"].cpu().numpy()  # (O, E)
    s_np, obs_np, te = s.cpu().numpy(), obs.cpu().numpy(), t_e.cpu().numpy()
    x_start = s_np[0, 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    x_lo, x_hi = obs_np[1, 0] - 0.05 * c * times[-1], obs_np[0, 0] + 0.05 * c * times[-1]
    half_height = 0.45 * c * times[-1]  # tall enough to show most fronts
    x_mid, half_width = 0.5 * (x_lo + x_hi), max(0.5 * (x_hi - x_lo), 1.75 * half_height)
    ax.set_xlim((x_mid - half_width) / 1000, (x_mid + half_width) / 1000)
    ax.set_ylim(-half_height / 1000, half_height / 1000)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    circles = [Circle((sx / 1000, 0), 0, fill=False, ec="#2563eb", lw=1.2) for sx in s_np[:, 0]]
    for circle in circles:
        ax.add_patch(circle)
    source = ax.plot([], [], "o", color="#1e40af", ms=8, zorder=5)[0]
    cone = [ax.plot([], [], color="#f59e0b", lw=2)[0] for _ in range(2)]
    labels = []
    for o, name in zip(obs_np, ["ahead", "behind"]):
        ax.plot([o[0] / 1000], [0], "^", color="#dc2626", ms=9, zorder=5)
        labels.append(
            ax.text(o[0] / 1000, 0.08 * half_height / 1000, "", ha="center", va="bottom", fontsize=8,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1})
        )  # fmt: skip
    title = ax.set_title("M = 0.00   source f = 0.00 Hz   t = 0.0 s")  # placeholder for tight_layout
    mach = v_s / c

    def update(step: int):
        t = times[step]
        for circle, r in zip(circles, radii[step]):
            circle.set_radius(r / 1000)
            circle.set_visible(r > 0)
        xs = x_start + v_s * t
        source.set_data([xs / 1000], [0])
        # Mach cone: two lines from the source, touching all fronts, half-angle μ with sin μ = c / v_s
        if mach > 1 and t > 0:
            mu = math.asin(1 / mach)
            length = v_s * t * math.cos(mu)  # distance from the source to where the cone touches the oldest front
            for sign, line in zip((1, -1), cone):
                line.set_data(
                    [xs / 1000, (xs - length * math.cos(mu)) / 1000], [0, sign * length * math.sin(mu) / 1000]
                )
        for label, name, arr in zip(labels, ["ahead", "behind"], arrivals):
            heard = arr[arr <= t]
            if len(heard) >= 2:
                label.set_text(f"{name}\nf' = {(len(heard) - 1) / (heard.max() - heard.min()):.2f} Hz")
            else:
                label.set_text(f"{name}\n(silent)")
        title.set_text(f"M = {mach:.2f}   source f = {1 / (te[1] - te[0]):.2f} Hz   t = {t:.1f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(times), 90), path, fps=12)
