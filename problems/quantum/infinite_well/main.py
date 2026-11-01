"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import norm


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n, L = params["n_grid"], params["L"]
    x, dx = tensors.grid(n, L, dev)
    H = tensors.hamiltonian(n, L, dev)

    energies, phi = solve.eigenstates(H, dx)
    if params["initial"] == "gaussian":
        c = solve.coefficients(phi, tensors.gaussian_packet(x, L, dx), dx)
        n_show = 3
    else:
        c = tensors.equal_superposition(params["n_states"], n, dev)
        n_show = max(3, params["n_states"])

    times, psi = solve.solve(phi, energies, c, params["revivals"] * solve.revival_time(L))

    for k, E in enumerate(energies[:4].tolist(), start=1):
        print(f"E_{k} = {E:9.4f}   exact n^2 pi^2/2L^2 = {(k * math.pi / L) ** 2 / 2:9.4f}")
    print(f"norm at t = 0: {norm(psi[0], dx):.6f}   at t_max: {norm(psi[-1], dx):.6f}")
    return plot.animate(x, times, psi, phi, n_show, L, out_dir / "infinite_well.gif")


if __name__ == "__main__":
    cli(run, __file__)
