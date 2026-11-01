"""Animate the water surface with its envelope; a red dot rides a crest (v_p), a green dot the envelope centre (v_g)."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif


def animate(times, x, a, xbar, k0: float, v_p: float, v_g: float, x0: float, sigma: float, path: Path) -> Path:
    t, xs = times.cpu().numpy(), x.cpu().numpy()
    eta, env = a.real.cpu().numpy(), a.abs().cpu().numpy()  # (N, J)
    centre = xbar.cpu().numpy()  # (N,)
    wavelength = 2 * math.pi / k0
    crest0 = wavelength * round(x0 / wavelength)  # a crest of cos(k0 x) near the packet centre at t = 0
    window = wavelength * math.ceil(1.5 * sigma / wavelength)  # keep the crest rider within ±window of the centre

    fig, ax = plt.subplots(figsize=(7, 3.8))
    line = ax.plot(xs, eta[0], color="#2563eb", lw=1.4)[0]
    env_hi = ax.plot(xs, env[0], "--", color="#9ca3af", lw=1)[0]
    env_lo = ax.plot(xs, -env[0], "--", color="#9ca3af", lw=1)[0]
    crest = ax.plot([], [], "o", color="#dc2626", ms=7, label=f"a crest, v_p = {v_p:.2f} m/s")[0]
    group = ax.plot([], [], "o", color="#16a34a", ms=7, label=f"envelope centre, v_g = {v_g:.2f} m/s")[0]
    x_hi = min(xs[-1], centre.max() + 4 * sigma + 20)
    ax.set_xlim(max(0.0, x0 - 4 * sigma - 20), x_hi)
    ax.set_ylim(-1.15, 1.45)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("surface η (relative)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    title = ax.set_title("deep-water wave packet   t = 0.0 s")

    def update(step: int):
        line.set_ydata(eta[step])
        env_hi.set_ydata(env[step])
        env_lo.set_ydata(-env[step])
        xc = crest0 + v_p * t[step]  # this crest moves at the phase velocity …
        xc = centre[step] + (xc - centre[step] + window) % (2 * window) - window  # … but jumps back whole wavelengths
        crest.set_data([xc], [np.interp(xc, xs, eta[step])])
        group.set_data([centre[step]], [np.interp(centre[step], xs, env[step])])
        title.set_text(f"deep-water wave packet   t = {t[step]:.1f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path, fps=12)
