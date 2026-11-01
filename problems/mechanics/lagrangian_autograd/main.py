"""Main loop: parameters → Lagrangian → autograd equations of motion → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    m, k, rest, g = params["m"], params["k"], params["rest_length"], 9.81
    L = equation.spring_pendulum(m, k, rest, g)
    y0 = tensors.initial_state(params["r0"], params["theta0"], dev)

    sol = solve.solve(y0, L, params["t_max"])

    E = sol["E"]
    hanging = rest + m * g / k  # equilibrium length of the hanging spring
    print(f"bounce frequency √(k/m) = {(k / m) ** 0.5:.3f} rad/s")
    print(f"swing frequency √(g/r_eq) ≈ {(g / hanging) ** 0.5:.3f} rad/s")
    print(f"energy: start {E[0]:.5f} J, relative drift {((E - E[0]) / E[0].abs()).abs().max():.1e}")
    return plot.animate(sol, rest, out_dir / "lagrangian_autograd.gif")


if __name__ == "__main__":
    cli(run, __file__)
