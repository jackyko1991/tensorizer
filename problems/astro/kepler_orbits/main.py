"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    GM = tensors.GM_SUN
    a, e = tensors.orbit_shapes(params["n_planets"], params["e_max"])
    y0 = tensors.initial_state(a, e, GM, dev)

    times, traj = solve.solve(y0, GM, params["t_max"], params["dt"])
    periods = solve.measured_periods(times, traj)

    E, L = equation.energy(traj, GM), equation.angular_momentum(traj)  # (T, B) each
    for b in range(len(a)):
        print(
            f"a = {a[b]:.2f} AU  e = {e[b]:.2f}  T = {periods[b]:.3f} yr  T²/a³ = {periods[b] ** 2 / a[b] ** 3:.4f}"
            f"  ΔE/E = {(E[:, b] / E[0, b] - 1).abs().max():.1e}  ΔL/L = {(L[:, b] / L[0, b] - 1).abs().max():.1e}"
        )
    return plot.animate(times, traj, a, e, periods, out_dir / "kepler_orbits.gif")


if __name__ == "__main__":
    cli(run, __file__)
