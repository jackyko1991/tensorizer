"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import probability


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    k0, V0, width = params["k0"], params["V0"], params["width"]
    x, k, dx = tensors.grid(2048, 200.0, dev)
    V = tensors.barrier(x, V0, width)
    psi0 = tensors.gaussian_packet(x, x0=-25.0, sigma=params["sigma"], k0=k0, dx=dx)

    times, psi = solve.solve(psi0, V, k, t_max=45.0 / k0)
    p_trans = solve.transmission(psi, x, width, dx)

    E = k0**2 / 2
    T_plane = solve.plane_wave_transmission(E, V0, tensors.grid_width(x, width, dx))
    print(f"packet energy E = {E:.3f}, barrier V0 = {V0:.3f}")
    print(f"transmission P_T = {p_trans[-1]:.4f}   plane-wave T(k0) = {T_plane:.4f}")
    print(f"norm at t_max: {probability(psi[-1], dx):.8f}")
    return plot.animate(x, times, psi, V, E, p_trans, out_dir / "tunneling.gif")


if __name__ == "__main__":
    cli(run, __file__)
