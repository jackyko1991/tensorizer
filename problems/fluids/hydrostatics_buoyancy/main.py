"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import pressure

DEPTH, P0, RHO_W, G = 1.0, 101_325.0, 1000.0, 9.81  # tank depth (m), air pressure (Pa), water density, gravity


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n, s = params["n_blocks"], params["size"]
    width = 0.35 * n
    g = tensors.gravity(G, dev)  # (2,)
    points = tensors.tank_points(width, DEPTH, 40, 60, dev)  # (nz, nx, 2)
    p = pressure(points, g, P0, RHO_W)  # (nz, nx)
    rho_b = tensors.densities(n, params["rho_min"], params["rho_max"], dev)  # (B,)
    y0 = tensors.initial_state(n, 0.15, dev)  # (B, 2)

    times, traj = solve.solve(y0, rho_b, s, DEPTH, params["t_max"], G, RHO_W, params["drag"])

    print(f"pressure at the floor: {p[0, 0]:.0f} Pa = p0 + rho g h = {P0 + RHO_W * G * DEPTH:.0f} Pa")
    for rho, f in zip(rho_b.tolist(), solve.floating_fraction(traj, s).tolist()):
        state = "floats" if rho < RHO_W else "sinks "
        print(f"rho = {rho:6.0f} kg/m³  {state}  submerged fraction {f:.3f}  (Archimedes {min(1, rho / RHO_W):.3f})")
    return plot.animate(times, traj, rho_b, p, width, DEPTH, s, RHO_W, out_dir / "hydrostatics_buoyancy.gif")


if __name__ == "__main__":
    cli(run, __file__)
