"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import effective_potential, hamiltonian, orbital_density, radial_probability

SPECTROSCOPIC = "spdf"


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_max = params["n_max"]
    r, dr = tensors.radial_grid(params["n_grid"], params["r_max"], dev)
    ls = tensors.angular_momenta(n_max, dev)  # (L,)
    H = hamiltonian(tensors.second_derivative(len(r), dr, dev), effective_potential(r, ls))  # (L, N, N)

    energies, u = solve.eigenstates(H, dr)

    frames = []
    for n, ell in solve.orbitals(n_max):
        E, u_nl = solve.pick(energies, u, n, ell)
        half = 3.0 * n**2 + 3.0  # slice size follows the orbital size ~ n²
        r_slice, cos_theta = tensors.slice_grid(half, 161, dev)
        density = orbital_density(solve.radial_on_slice(u_nl, dr, r_slice), ell, cos_theta)  # (P, P)
        nodes = solve.count_nodes(u_nl)
        name = f"{n}{SPECTROSCOPIC[ell]}"
        print(f"{name}:  E = {E:9.5f}   exact -1/(2n^2) = {-0.5 / n**2:9.5f}   nodes = {nodes} (n-l-1 = {n - ell - 1})")
        frames.append(
            dict(name=name, n=n, l=ell, E=E, nodes=nodes, P=radial_probability(u_nl), density=density, half=half)
        )
    return plot.animate(r, frames, out_dir / "hydrogen_radial.gif")


if __name__ == "__main__":
    cli(run, __file__)
