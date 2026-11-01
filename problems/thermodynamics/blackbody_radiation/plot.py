"""Animate the spectrum while the temperature sweeps up, beside the total energy and the T⁴ law."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import WIEN_B, stefan_boltzmann

LAM_SHOWN = (100, 3000)  # wavelength window of the plot (nm)


def animate(T: torch.Tensor, lam: torch.Tensor, r: dict, path: Path) -> Path:
    temps = T.cpu().numpy()
    lam_nm = 1e9 * lam.cpu().numpy()  # (L,)
    u = 1e-6 * r["u"].cpu().numpy()  # (A, L)  J/(m³ µm)
    lam_max = 1e9 * r["lam_max"].cpu().numpy()  # (A,)
    U = r["U"].cpu().numpy()  # (A,)
    rgb = r["rgb"].cpu().numpy()  # (A, 3)
    T_line = torch.linspace(temps[0], temps[-1], 100, dtype=torch.float64)

    fig, (ax, ax_U) = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: the spectrum, coloured by its approximate glow, with the visible band shaded.
    ax.axvspan(380, 750, color="white", alpha=0.15, lw=0, label="visible")
    curve = ax.plot([], [], lw=2.5)[0]
    fill = [ax.fill_between(lam_nm, 0, 0)]
    peak = ax.plot([], [], "v", color="black", ms=7, label="Wien peak")[0]
    ax.set_xlim(*LAM_SHOWN)
    ax.set_xlabel("wavelength λ (nm)")
    ax.set_ylabel("u(λ, T)  (J m⁻³ µm⁻¹)")
    ax.legend(fontsize=8, loc="upper right")
    title = ax.set_title("")
    ax.set_facecolor("#1f2937")  # dark background so the glow colour stands out
    ax.grid(color="#374151")

    # Right: total energy U = ∫u dλ against T, with the exact 4σT⁴/c.
    ax_U.plot(T_line.numpy(), stefan_boltzmann(T_line).numpy(), color="gray", ls="--", label="4σT⁴/c")
    U_line = ax_U.plot([], [], color=ACCENT[1], lw=2, label="∫ u dλ (trapezoid)")[0]
    U_dot = ax_U.plot([], [], "o", color=ACCENT[1])[0]
    ax_U.set_xlim(temps[0], temps[-1])
    ax_U.set_ylim(0, 1.1 * U.max())
    ax_U.set_xlabel("T (K)")
    ax_U.set_ylabel("U (J/m³)")
    ax_U.set_title("Stefan–Boltzmann: U ∝ T⁴")
    ax_U.legend(fontsize=8, loc="upper left")

    def update(step: int):
        color = tuple(rgb[step])
        curve.set_data(lam_nm, u[step])
        curve.set_color(color)
        fill[0].remove()
        fill[0] = ax.fill_between(lam_nm, 0, u[step], color=color, alpha=0.25, lw=0)
        top = u[step].max()
        peak.set_data([lam_max[step]], [top * 1.04])
        ax.set_ylim(0, 1.15 * top)  # the peak grows like T⁵, so rescale every frame
        title.set_text(
            f"T = {temps[step]:.0f} K   λ_max = {lam_max[step]:.0f} nm   (b / T = {1e9 * WIEN_B / temps[step]:.0f})"
        )
        U_line.set_data(temps[: step + 1], U[: step + 1])
        U_dot.set_data([temps[step]], [U[step]])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(temps)), path)
