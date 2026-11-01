"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import cyclotron_frequency, drift_velocity


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    eps = tensors.levi_civita(dev)
    E, B = tensors.fields(params["E"], params["B"], dev)
    q, m = tensors.particles(params["n_particles"], dev)
    y0 = tensors.initial_state(params["n_particles"], params["v_par"], dev)

    times, traj = solve.solve(y0, q, m, E, B, eps, params["t_max"])

    drift = drift_velocity(E, B, eps)
    for qb, mb, w in zip(q.tolist(), m.tolist(), cyclotron_frequency(q, m, B).tolist()):
        print(f"q {qb:+.0f}  m {mb:3.1f}  cyclotron omega = {w:.3f}")
    print(f"ExB drift velocity: {[round(u, 3) for u in drift.tolist()]}")
    return plot.animate(times, traj, q, m, drift, out_dir / "lorentz_force.gif")


if __name__ == "__main__":
    cli(run, __file__)
