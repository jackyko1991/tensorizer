"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import mean_and_width, norm


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    k0, sigma0 = params["k0"], params["sigma0"]
    x0 = -40.0
    x, k, dx = tensors.grid(4096, 300.0, dev)
    psi0 = tensors.gaussian_packet(x, x0, sigma0, k0, dx)

    times, psi = solve.solve(psi0, k, params["t_max"])
    mean, width = mean_and_width(psi, x, dx)
    mean_exact = solve.exact_centre(times, x0, k0)
    width_exact = solve.exact_width(times, sigma0)

    print(f"<x>(t_max) = {mean[-1]:8.3f}   exact x0 + k0 t = {mean_exact[-1]:8.3f}")
    print(f"sigma(t_max) = {width[-1]:8.3f}   exact sigma0 sqrt(1 + (t/2sigma0^2)^2) = {width_exact[-1]:8.3f}")
    print(f"norm at t_max: {norm(psi[-1], dx):.10f}")
    return plot.animate(x, times, psi, mean_exact, width_exact, out_dir / "free_wave_packet.gif")


if __name__ == "__main__":
    cli(run, __file__)
