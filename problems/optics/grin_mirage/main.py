"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dn, h = params["delta_n"] * 1e-5, params["scale_height"]
    eye, distance = params["eye_height"], params["distance"]
    angles = tensors.viewing_angles(params["max_dip"], params["n_rays"])
    state0 = tensors.initial_state(angles, eye, distance, dn, h)

    path = solve.solve(state0, dn, h, distance)
    seen = solve.height_at_object(path)

    for a, y in zip((angles * 1e3).tolist(), seen.tolist()):
        target = "road" if math.isnan(y) else f"object at {y:.2f} m"
        print(f"looking {a:+6.2f} mrad  ->  {target}")
    return plot.animate(path, angles, seen, eye, h, out_dir / "grin_mirage.gif")


if __name__ == "__main__":
    cli(run, __file__)
