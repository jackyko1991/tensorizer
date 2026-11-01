"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import plot, solve, tensors
from .equation import field_tensor


def run(params: dict, out_dir: Path) -> Path:
    E, B = tensors.fields(params)  # (3,), (3,)
    F = field_tensor(E, B, tensors.levi_civita(3))  # (4, 4)   tiny tensors: CPU float64

    sweep = solve.solve(F, params["beta_max"])

    for key in ["E", "B"]:
        print(f"{key} at beta = 0: {sweep[key][0].tolist()}  ->  at beta_max: {sweep[key][-1].numpy().round(3)}")
    print(f"F_mn F^mn = {sweep['FF'][0]:.4f} -> {sweep['FF'][-1]:.4f}   (invariant)")
    return plot.animate(sweep, out_dir / "field_tensor_boost.gif")


if __name__ == "__main__":
    cli(run, __file__)
