"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_layer = tensors.indices(params["n1"], params["n2"], params["n3"], dev)
    angles = tensors.fan_angles(params["n_rays"], params["max_angle"])
    p0, d0 = tensors.initial_rays(angles, dev)

    times, rays = solve.solve(p0, d0, n_layer, params["t_max"])

    for a, b in zip(n_layer[:-1].tolist(), n_layer[1:].tolist()):
        theta_c = solve.critical_angle(a, b)
        note = f"critical angle {theta_c:.1f}°" if theta_c else "no total internal reflection"
        print(f"interface n = {a:.2f} -> {b:.2f}: {note}")
    return plot.animate(times, rays, n_layer, angles, out_dir / "snell_refraction.gif")


if __name__ == "__main__":
    cli(run, __file__)
