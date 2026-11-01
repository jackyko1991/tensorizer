"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    _, points = tensors.grid(params["n_grid"], dev)  # (G, 2) arrows
    line = tensors.x_axis(300, dev)  # (300, 2) points along y = 0
    charges = tensors.charges(params["q_fixed"], params["q_moving"], dev)
    path = tensors.charge_path(72, dev)  # (T, 2, 2)

    E = solve.solve(points, path, charges)  # (T, G, 2)
    E_line = solve.solve(line, path, charges)  # (T, 300, 2)
    E_parts = solve.each_charge(line, path, charges)  # (T, 2, 300, 2)

    print(f"{len(points)} grid points x {len(charges)} charges x {len(path)} positions")
    print(f"superposition check: max |E - sum of parts| = {(E_line - E_parts.sum(1)).abs().max():.2e}")
    return plot.animate(points, E, line, E_line, E_parts, path, charges, out_dir / "coulomb_field.gif")


if __name__ == "__main__":
    cli(run, __file__)
