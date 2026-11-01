"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    eps = tensors.levi_civita(dev)  # (3, 3, 3)
    radii = tensors.radii(params["radius"], 72)

    result = solve.solve(params["layout"], radii, params["current"], params["n_segments"], params["n_grid"], eps, dev)

    error = (result["B_axis"][..., 2] - result["B_axis_exact"]).abs().max() / result["B_axis_exact"].abs().max()
    print(f"{params['n_segments']} segments per loop, {len(result['plane'])} field points per frame")
    print(f"centre field at R = {radii[0]:.2f} m: {result['B_axis_exact'][0].abs().max() * 1e6:.2f} μT")
    print(f"largest axis error, segments vs exact formula: {error:.2e} (relative)")
    return plot.animate(result, params["layout"], out_dir / "biot_savart.gif")


if __name__ == "__main__":
    cli(run, __file__)
