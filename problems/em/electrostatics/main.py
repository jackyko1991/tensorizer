"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    n = params["n_grid"]
    x, h = tensors.grid(n)
    rho = tensors.charge_density(params["config"], n, params["charge"])  # CPU float64: a direct solve

    phi, E = solve.solve(rho, h)

    print(f"grid {n}x{n}: {n * n} unknowns, potential range [{phi.min():.3f}, {phi.max():.3f}]")
    return plot.draw(x, phi, E, rho, out_dir / "electrostatics.png")


if __name__ == "__main__":
    cli(run, __file__)
