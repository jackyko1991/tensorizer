"""Animate the cloud of walkers over the Gaussian density, next to the growing MSD and Einstein's line."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from .equation import gaussian_density, msd_theory

N_SHOWN = 400  # dots drawn (all walkers are used for the MSD)
N_TRAILS = 4  # walkers whose whole path is drawn


def animate(t: torch.Tensor, x: torch.Tensor, msd: torch.Tensor, D: float, D_fit: float, path: Path) -> Path:
    times = t.cpu().numpy()
    pos = x[:, :N_SHOWN].cpu().numpy()  # (T, N_SHOWN, 2)
    msd_np = msd.cpu().numpy()
    half = 3.0 * (2 * D * times[-1]) ** 0.5  # ±3 standard deviations at the end
    grid = torch.linspace(-half, half, 120)
    X, Y = torch.meshgrid(grid, grid, indexing="xy")  # (G, G)

    fig, (ax, ax_m) = plt.subplots(1, 2, figsize=(7, 4))
    density = ax.imshow(
        torch.zeros(120, 120).numpy(), extent=(-half, half, -half, half), origin="lower", cmap="Blues", vmin=0
    )
    dots = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=2, color="black", alpha=0.5)
    trails = [ax.plot([], [], lw=1, color=ACCENT[1 + k])[0] for k in range(N_TRAILS)]
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    clock = ax.set_title("")

    ax_m.plot(times, msd_theory(t, D).cpu().numpy(), color="gray", ls="--", label=f"2dDt  (D = {D:.2f})")
    msd_line = ax_m.plot([], [], color=ACCENT[0], lw=2, label=f"simulation  (fit D = {D_fit:.2f})")[0]
    ax_m.set_xlim(0, times[-1])
    ax_m.set_ylim(0, 1.15 * max(msd_np.max(), 4 * D * times[-1]))
    ax_m.set_xlabel("t (s)")
    ax_m.set_ylabel("⟨x_i x_i⟩ (µm²)")
    ax_m.set_title("mean-squared displacement")
    ax_m.legend(fontsize=8, loc="upper left")

    def update(step: int):
        if step > 0:  # the density is a point (delta function) at t = 0
            rho = gaussian_density(X, Y, float(times[step]), D)
            density.set_data(rho.numpy())
            density.set_clim(0, rho.max().item())
        dots.set_offsets(pos[step])
        for k, trail in enumerate(trails):
            trail.set_data(pos[: step + 1, k, 0], pos[: step + 1, k, 1])
        msd_line.set_data(times[: step + 1], msd_np[: step + 1])
        clock.set_text(f"t = {times[step]:.1f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(times), 72), path)
