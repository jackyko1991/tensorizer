"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n, k, m = params["n_masses"], params["k"], params["m"]
    K = tensors.stiffness(n, k, dev)
    x0 = tensors.initial_displacement(n, params["start"], params["amplitude"], dev)

    sol = solve.solve(K, m, x0, params["t_max"])
    kinetic, potential = equation.energy(sol["x"], sol["v"], K, m)

    exact = tensors.analytic_frequencies(n, k, m).tolist()
    for p, (w, a) in enumerate(zip(sol["omega"].tolist(), sol["A"].tolist())):
        print(f"mode {p + 1:2d}  omega = {w:6.3f} rad/s  (analytic {exact[p]:6.3f})  amplitude {a:6.3f} m")
    return plot.animate(sol, kinetic, potential, out_dir / "coupled_oscillators.gif")


if __name__ == "__main__":
    cli(run, __file__)
