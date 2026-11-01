"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    n, T_h, T_c = params["n"], params["T_h"], params["T_c"]
    gamma = tensors.GAMMA[params["gas"]]
    legs = tensors.cycle(n, T_h, T_c, params["ratio"], gamma)  # (4, K, 2)

    result = solve.solve(legs, n, T_h, T_c, gamma)
    eta, carnot = result["efficiency"], 1 - T_c / T_h
    for name, w in zip(plot.LEG_NAMES, result["W_legs"].tolist()):
        print(f"{name:22s} W = {w:9.1f} J")
    print(f"W = {result['W']:.1f} J   Q_h = {result['Q_h']:.1f} J   Q_c = {result['Q_c']:.1f} J")
    print(f"efficiency W / Q_h = {eta:.4f}   1 - T_c / T_h = {carnot:.4f}")

    title = f"W = {result['W']:.0f} J,  Q_h = {result['Q_h']:.0f} J,  η = W/Q_h = {eta:.3f} = 1 − T_c/T_h"
    return plot.animate(legs, result["T"], result["S"], title, out_dir / "carnot_cycle.gif")


if __name__ == "__main__":
    cli(run, __file__)
