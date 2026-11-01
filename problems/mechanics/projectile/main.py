"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    g = tensors.gravity(dev)
    angles = tensors.launch_angles(params["n_angles"])
    y0 = tensors.initial_state(params["v0"], angles, dev)

    times, traj = solve.solve(y0, g, params["drag"], params["t_max"])

    for deg, r in zip(angles.rad2deg().tolist(), solve.ranges(traj).tolist()):
        print(f"angle {deg:5.1f}°  range {r:7.2f} m")
    return plot.animate(times, traj, angles, out_dir / "projectile.gif")


if __name__ == "__main__":
    cli(run, __file__)
