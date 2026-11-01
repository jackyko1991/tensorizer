"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n, spacing, node_mass = params["n_side"], 0.1, 0.01
    x0 = tensors.grid(n, spacing, dev)  # (N, 3)
    edges = tensors.springs(n, dev)  # (E, 2)
    D = tensors.incidence(edges, n * n)  # (E, N)
    rest = tensors.rest_lengths(x0, edges)  # (E,)
    free = tensors.free_mask(n, dev)  # (N,)
    cloth = tensors.parameters(params["k"], params["damping"], node_mass, 9.81, dev)

    dt = min(0.002, 0.15 * math.sqrt(node_mass / params["k"]))  # stable RK4 step for the stiffest spring
    times, traj = solve.solve(x0, D, rest, free, cloth, params["t_max"], dt)
    E = equation.energy(traj, D, rest, cloth)  # (T,)

    print(f"{n * n} masses, {len(edges)} springs, dt = {dt * 1e3:.2f} ms, {len(times) - 1} steps")
    print(f"energy: start {E[0]:.4f} J, end {E[-1]:.4f} J")
    return plot.animate(times, traj, E, n, out_dir / "mass_spring_cloth.gif")


if __name__ == "__main__":
    cli(run, __file__)
