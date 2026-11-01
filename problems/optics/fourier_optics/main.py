"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    size_min, size_max = params["size_min"], max(params["size_min"], params["size_max"])
    schedule = tensors.schedule(size_min, size_max, params["frames_per_aperture"])

    masks, patterns = solve.solve(schedule)

    for d in (size_min, size_max):
        measured, predicted = solve.first_dark_ring(d), equation.airy_radius(tensors.N, d)
        print(f"circle D = {d:2d} px: first dark ring at {measured:.0f} px (1.22 N/D = {predicted:.1f} px)")
    return plot.animate(schedule, masks, patterns, out_dir / "fourier_optics.gif")


if __name__ == "__main__":
    cli(run, __file__)
