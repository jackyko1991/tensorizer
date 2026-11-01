"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    s, f1, f2 = params["s"], params["f1"], params["f2"]
    rays = tensors.object_rays(params["h0"], params["half_angle"], params["n_rays"])
    gaps = tensors.separations(params["d_start"], params["d_end"])
    z = tensors.axis(s + params["d_end"] + 60.0)

    heights, image_z, image_h = solve.solve(rays, s, f1, f2, gaps, z)

    m = equation.system_matrix(s, f1, gaps[0].item(), f2)
    print(f"system matrix at d = {gaps[0]:.1f} cm:\n{m}\ndet M = {m.det():.6f}")
    return plot.animate(z, heights, image_z, image_h, gaps, s, params["h0"], out_dir / "lens_system.gif")


if __name__ == "__main__":
    cli(run, __file__)
