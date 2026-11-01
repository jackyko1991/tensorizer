"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    radius, n_rays, aperture = params["radius"], params["n_rays"], params["aperture"]
    lamp = tensors.lamp_rays(radius, n_rays, aperture)
    parallel = tensors.parallel_rays(radius, n_rays, aperture)
    flat = tensors.flat_mirror()
    centre = tensors.mirror_centre(radius)
    times = tensors.times(radius)

    result = solve.solve(radius, lamp, parallel, flat, centre, times)

    print(f"focal point F = R/2 at x = {-radius / 2:.2f} m")
    for y, x in zip(parallel[0][:, 1].tolist(), result["crossings"].tolist()):
        if y > 0:
            print(f"ray at height {y:5.2f} m crosses the axis at x = {x:6.3f} m")
    return plot.animate(times, result, radius, centre, out_dir / "mirror_reflection.gif")


if __name__ == "__main__":
    cli(run, __file__)
