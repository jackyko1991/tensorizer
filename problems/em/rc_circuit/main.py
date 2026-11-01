"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import stored_energy


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    C = params["C"]
    R = tensors.resistances(params["R_min"], params["R_max"], params["n_circuits"], dev)  # (B,)
    V0 = tensors.initial_voltage(len(R), dev)  # (B,)

    result = solve.solve(V0, R, C, params["V_s"], params["source"], params["period"], params["t_max"])

    U = stored_energy(result["V"][-1], C)  # (B,)  kΩ·mF = s, V²·mF = mJ
    for r, v, u in zip(R.tolist(), result["V"][-1].tolist(), U.tolist()):
        print(f"R {r:5.2f} kΩ  τ = {r * C:5.2f} s   V(end) = {v:5.2f} V   ½CV² = {u:6.2f} mJ")
    return plot.animate(result, R, C, out_dir / "rc_circuit.gif")


if __name__ == "__main__":
    cli(run, __file__)
