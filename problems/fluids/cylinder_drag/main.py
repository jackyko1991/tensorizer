"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import viscosity_to_tau

NX, NY, D, U_IN = 180, 56, 10, 0.12  # lattice size, cylinder diameter (nodes), inflow speed (lattice units)
CENTRE = (40.0, NY / 2)


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    re, n_steps = params["re"], params["steps"]
    nu = U_IN * D / re  # lattice viscosity from Re = U D / ν
    tau = viscosity_to_tau(nu)
    e, w, opp = tensors.lattice(dev)  # (9, 2), (9,), (9,)
    solid = tensors.cylinder(NX, NY, *CENTRE, D / 2, dev)  # (Nx, Ny) bool
    links = tensors.boundary_links(solid, e)  # (Nx, Ny, 9)
    f = tensors.initial_populations(NX, NY, U_IN, e, w, dev)  # (Nx, Ny, 9)

    result = solve.simulate(f, solid, links, e, w, opp, tau, U_IN, n_steps, save_every=max(1, n_steps // 100))
    coeffs = solve.coefficients(result["force"], U_IN, D)  # (T, 2): C_D, C_L
    late = coeffs[n_steps // 2 :]

    print(
        f"lattice {NX}x{NY}, D = {D}, U = {U_IN}, Re = {re:.0f}, nu = {nu:.4f}, tau = {tau:.3f}, "
        f"blockage D/H = {D / NY:.2f}"
    )
    print(f"second half of the run: mean C_D = {late[:, 0].mean():.2f}, mean C_L = {late[:, 1].mean():+.3f}, "
          f"C_L amplitude {late[:, 1].abs().max():.2f}")  # fmt: skip
    print(
        f"Strouhal number St = f D / U = {solve.strouhal(coeffs[:, 1], U_IN, D):.3f}  (free cylinder at Re 100: ≈ 0.16)"
    )
    return plot.animate(
        result["vorticity"], result["steps"], coeffs, CENTRE, D / 2, U_IN, out_dir / "cylinder_drag.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
