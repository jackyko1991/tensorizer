"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    n, T0 = params["n"], params["T0"]
    gamma = tensors.GAMMA[params["gas"]]
    paths = tensors.processes(n, T0, params["ratio"], gamma)  # (4, K, 2)

    r = solve.solve(paths, n, gamma)
    for c, name in enumerate(tensors.NAMES):
        print(
            f"{name:10s}  W = {r['W'][c]:8.1f} J   ΔU = {r['dU'][c]:8.1f} J   Q = {r['Q'][c]:8.1f} J"
            f"   T_end = {r['T'][c, -1]:6.1f} K"
        )
    return plot.animate(paths, r, tensors.NAMES, out_dir / "ideal_gas_processes.gif")


if __name__ == "__main__":
    cli(run, __file__)
