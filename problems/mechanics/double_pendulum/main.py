"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    m1, m2, L1, L2, g = params["m1"], params["m2"], params["L1"], params["L2"], 9.81
    y0 = tensors.initial_state(params["theta1"], params["theta2"], params["delta"], dev)

    times, traj = solve.solve(y0, m1, m2, L1, L2, g, params["t_max"])
    e = equation.energy(traj, m1, m2, L1, L2, g)  # (T, B)

    omega, _ = equation.small_angle_modes(m1, m2, L1, L2, g)
    print(f"small-angle normal modes: ω = {omega[0]:.3f}, {omega[1]:.3f} rad/s")
    print(f"relative energy drift: {((e[-1] - e[0]) / e[0].abs()).abs().max():.2e}")
    gap = (traj[:, 0, 0] - traj[:, 1, 0]).abs().max(dim=-1).values  # (T,) largest angle difference
    print(f"angle difference: start {gap[0]:.1e} rad, end {gap[-1]:.2f} rad")
    return plot.animate(times, traj, L1, L2, out_dir / "double_pendulum.gif")


if __name__ == "__main__":
    cli(run, __file__)
