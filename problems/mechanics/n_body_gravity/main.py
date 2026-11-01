"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    G, eps = tensors.G, params["softening"]
    m, x0, v0 = tensors.initial_state(params["scenario"], dev)

    times, xs, vs = solve.solve(m, x0, v0, G, eps, params["t_max"], params["dt"])

    e_start = equation.energy(xs[0], vs[0], m, G, eps).item()
    e_end = equation.energy(xs[-1], vs[-1], m, G, eps).item()
    p_end = equation.momentum(vs[-1], m).norm().item()
    print(f"{len(m)} bodies, {len(times) - 1} steps")
    print(f"energy {e_start:.6f} -> {e_end:.6f}  (relative change {abs(e_end - e_start) / abs(e_start):.2e})")
    print(f"total momentum at the end |P| = {p_end:.2e}")
    return plot.animate(times, xs, m, out_dir / "n_body_gravity.gif")


if __name__ == "__main__":
    cli(run, __file__)
