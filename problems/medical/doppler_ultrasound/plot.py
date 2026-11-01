"""Animate the vessel with moving scatterers and the transducer beam, next to the spectrogram as it builds up."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Arc

from problems._common.plotting import frame_indices, save_gif

from .equation import doppler_shift

SLOW = 0.1  # the animation shows the cells moving 10× slower than real time


def animate(
    t: torch.Tensor,
    v: torch.Tensor,
    v_max: torch.Tensor,
    scat: dict[str, torch.Tensor],
    win_t: torch.Tensor,
    freqs: torch.Tensor,
    power: torch.Tensor,
    f0: float,
    theta: float,
    prf: float,
    length: float,
    path: Path,
) -> Path:
    dt = 1.0 / prf
    times = t.cpu().numpy()
    frames = frame_indices(len(times), 75)
    # displacement of every cell (mm) at the rendered frames only: running integral of its velocity
    x_mm = (scat["x"] + SLOW * 1e3 * torch.cumsum(v, dim=0) * dt)[frames]  # (frames, N)
    x_mm = torch.remainder(x_mm, length).cpu().numpy()
    speed = v[frames].abs().cpu().numpy()  # (frames, N)
    y_mm = (scat["y"] * 1e3).cpu().numpy()
    R_mm = float(scat["r"].max()) * 1e3 * 1.02

    fig, (ax, ax_s) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1, 1.45]})

    # --- left: longitudinal view of the vessel, transducer and beam -----------------------------------
    ax.set_xlim(-1, length + 1)
    ax.set_ylim(-R_mm - 1.5, 13)
    ax.set_aspect("equal")
    ax.set_xlabel("along vessel (mm)")
    ax.set_ylabel("mm")
    ax.grid(False)
    ax.fill_between([-1, length + 1], -R_mm, R_mm, color="#fee2e2", zorder=0)
    for y in (-R_mm, R_mm):
        ax.axhline(y, color="#374151", lw=2)
    centre = np.array([length / 2, 0.0])
    beam = np.array([math.cos(theta), math.sin(theta)])  # from the vessel centre towards the transducer
    probe = centre + 10.5 * beam
    side = np.array([-beam[1], beam[0]])
    ax.plot(*np.stack([probe - 2.2 * side, probe + 2.2 * side]).T, color="#1e40af", lw=7, solid_capstyle="butt")
    ax.plot(*np.stack([probe - 0.6 * beam, centre - (R_mm / max(beam[1], 0.3)) * beam]).T, "--", color="#f59e0b")
    ax.add_patch(Arc(centre, 6, 6, theta1=0, theta2=math.degrees(theta), color="#f59e0b", lw=1.5))
    ax.text(*(centre + 3.8 * np.array([math.cos(theta / 2), math.sin(theta / 2)])), "θ", color="#b45309")
    ax.text(probe[0] + 1.0, probe[1] + 1.2, f"{f0 / 1e6:.0f} MHz", fontsize=8, color="#1e40af")
    ax.annotate("", xy=(4.5, -R_mm - 0.9), xytext=(0.5, -R_mm - 0.9), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(5.0, -R_mm - 1.3, "flow (shown 10× slower)", fontsize=7)
    vmax = float(v_max.abs().max())
    dots = ax.scatter(x_mm[0], y_mm, c=speed[0], s=9, cmap="plasma", vmin=0, vmax=vmax, zorder=3)
    cb = fig.colorbar(dots, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("speed (m/s)")
    title = ax.set_title("t = 0.00 s")

    # --- right: Doppler spectrogram (frequency or velocity vs time) -------------------------------------
    db = 10 * torch.log10(power / power.max() + 1e-12).clamp(min=-40).cpu().numpy()  # (F, K)
    wt = win_t.cpu().numpy()
    f_khz = freqs.cpu().numpy() / 1e3
    df = (f_khz[1] - f_khz[0]) / 2
    cmap = plt.get_cmap("magma", 16).with_extremes(bad="white")
    image = ax_s.imshow(
        np.full_like(db, np.nan),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-40,
        vmax=0,
        extent=[wt[0], wt[-1], f_khz[0] - df, f_khz[-1] + df],
        interpolation="nearest",
    )
    expected = doppler_shift(v_max, f0, theta).cpu().numpy() / 1e3
    ax_s.plot(times, expected, "--", color="#22d3ee", lw=1, label="$f_D$ of centreline $v_{\\max}(t)$")
    ax_s.set_ylim(-prf / 2e3, prf / 2e3)
    ax_s.set_xlim(0, times[-1])
    ax_s.set_xlabel("time (s)")
    ax_s.set_ylabel("Doppler shift (kHz)")
    ax_s.set_title(f"spectrogram, Nyquist ±PRF/2 = ±{prf / 2e3:.1f} kHz")
    ax_s.legend(loc="lower right", fontsize=7)
    ax_s.grid(False)
    if math.cos(theta) > 0.05:
        khz_per_ms = float(doppler_shift(torch.tensor(1.0), f0, theta)) / 1e3  # kHz of shift per m/s of flow
        velocity_axis = ax_s.secondary_yaxis("right", functions=(lambda f: f / khz_per_ms, lambda u: u * khz_per_ms))
        velocity_axis.set_ylabel("velocity (m/s)")
    cursor = ax_s.axvline(0, color="#9ca3af", lw=1)

    def update(k: int):
        step = frames[k]
        dots.set_offsets(np.column_stack([x_mm[k], y_mm]))
        dots.set_array(speed[k])
        shown = np.where(wt[None, :] <= times[step], db, np.nan)
        image.set_data(shown)
        cursor.set_xdata([times[step]])
        title.set_text(f"t = {times[step]:.2f} s   $v_{{\\max}}$ = {float(v_max[step]):.2f} m/s")

    fig.tight_layout()
    return save_gif(fig, update, list(range(len(frames))), path, fps=15)
