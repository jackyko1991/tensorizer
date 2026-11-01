"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import divergence, laplacian, velocity

LID = 1.0  # lid speed; with cavity size 1 the Reynolds number is Re = U L / ν = 1/ν


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n, re = params["n"], params["re"]
    nu = LID / re
    y, h = tensors.grid(n, dev)  # (N,)
    V, lam = tensors.laplacian_eigen(n - 2, h, dev)  # (n, n), (n,)
    omega, _ = tensors.at_rest(n, dev)  # (N, N)

    times, omegas, psis = solve.simulate(omega, V, lam, h, nu, LID, params["t_max"], save_every=10)
    u_centre = torch.stack([solve.centreline_u(p, h) for p in psis])  # (F, N)

    psi = psis[-1]
    u = velocity(psi, h)  # (2, N-2, N-2)
    residual = laplacian(psi, h) + omegas[-1][1:-1, 1:-1]
    i, j = divmod(int(psi.argmin()), n)
    print(f"{n}x{n} grid, Re = {re:.0f}, dt = {solve.time_step(h, nu, LID):.4f}, t = {times[-1]:.1f}")
    print(
        f"Poisson residual max |∇²ψ + ω| = {residual.abs().max():.1e};  "
        f"max |∂_i u_i| = {divergence(u, h).abs().max():.1e}"
    )
    print(
        f"primary vortex centre ({y[i]:.3f}, {y[j]:.3f}), ψ_min = {psi.min():.4f}  "
        "(Ghia Re=100: (0.617, 0.734), -0.1034)"
    )
    ghia = None
    if abs(re - 100) < 1e-9:
        ghia = solve.GHIA_RE100
        print(f"centreline u vs Ghia et al.: max deviation {solve.ghia_deviation(y, u_centre[-1]):.3f}")
    return plot.animate(y, times, omegas, psis, u_centre, ghia, re, out_dir / "lid_driven_cavity.gif")


if __name__ == "__main__":
    cli(run, __file__)
