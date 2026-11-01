"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import mean_position, norm


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    omega, x0 = params["omega"], params["x0"]
    L = 2 * (x0 + 7 / math.sqrt(omega))  # box much wider than the swing plus a few ground-state widths
    x, dx = tensors.grid(params["n_grid"], L, dev)
    H = tensors.hamiltonian(x, dx, omega)

    energies, phi = solve.eigenstates(H, dx)
    c = solve.coefficients(phi, tensors.coherent_state(x, x0, omega), dx)
    period = 2 * math.pi / omega
    times, psi = solve.solve(phi, energies, c, params["periods"] * period)

    for n, E in enumerate(energies[:4].tolist()):
        print(f"E_{n} = {E:8.4f}   exact omega*(n + 1/2) = {omega * (n + 0.5):8.4f}")
    x_mean = mean_position(psi, x, dx)
    t_end = times[-1].item()
    print(f"<x> at t_max = {x_mean[-1].item():.4f}   classical x0 cos(wt) = {x0 * math.cos(omega * t_end):.4f}")
    print(f"norm at t_max: {norm(psi[-1], dx):.8f}")
    return plot.animate(x, times, psi, phi, energies, x_mean, omega, x0, out_dir / "harmonic_oscillator.gif")


if __name__ == "__main__":
    cli(run, __file__)
