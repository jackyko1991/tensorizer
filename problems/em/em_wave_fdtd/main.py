"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import courant_time_step

NX, NY, BORDER = 200, 130, 20  # grid cells and width of the absorbing border; cell size h = 1


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    eps = tensors.levi_civita(dev)  # (3, 3, 3)
    eps_r = tensors.permittivity(NX, NY, params["eps_block"], dev)  # (Nx, Ny)
    damping = tensors.damping(NX, NY, BORDER, 0.1, dev)  # (Nx, Ny)
    E, H = tensors.empty_fields(NX, NY, dev)  # (3, Nx, Ny) each
    mask = solve.point_source(NX, NY, dev) if params["source"] == "point" else solve.line_source(NX, NY, dev)

    dt = courant_time_step(1.0, params["courant"], dim=2)
    n_steps = int(params["t_max"] / dt)
    waveform = solve.continuous_wave(params["wavelength"])
    times, Ez = solve.simulate(E, H, eps_r, damping, mask, waveform, eps, dt, 1.0, n_steps, save_every=4)

    n_block = params["eps_block"] ** 0.5
    print(f"grid {NX}x{NY}, dt = {dt:.3f} (Courant number {params['courant']}), {n_steps} steps")
    reflection = (1 - n_block) / (1 + n_block)  # amplitude reflected at normal incidence
    print(
        f"block: n = {n_block:.2f}, wavelength {params['wavelength']:.0f} -> {params['wavelength'] / n_block:.1f} cells"
    )
    print(f"normal-incidence reflection coefficient r = (1 - n)/(1 + n) = {reflection:+.2f}")
    return plot.animate(times, Ez, eps_r, BORDER, out_dir / "em_wave_fdtd.gif")


if __name__ == "__main__":
    cli(run, __file__)
