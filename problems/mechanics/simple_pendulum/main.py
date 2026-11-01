"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    g, L = params["g"], params["length"]
    theta0 = tensors.amplitudes(params["n_pendulums"], params["max_amplitude"])
    y0 = tensors.initial_state(theta0, dev)

    times, traj = solve.solve(y0, g, L, params["t_max"])
    small = equation.small_angle(y0[:, 0], times, g, L)  # (T, B)

    measured = solve.periods(times, traj[:, :, 0])
    exact = equation.exact_period(y0[:, 0], g, L)
    print(f"small-angle period 2π√(L/g) = {2 * math.pi * math.sqrt(L / g):.3f} s")
    for deg, tm, te in zip(theta0.rad2deg().tolist(), measured.tolist(), exact.tolist()):
        print(f"amplitude {deg:5.1f}°  period {tm:6.3f} s  (elliptic integral {te:6.3f} s)")
    return plot.animate(times, traj, small, L, out_dir / "simple_pendulum.gif")


if __name__ == "__main__":
    cli(run, __file__)
