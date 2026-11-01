"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import entanglement_entropy, reduced_density_1, schmidt


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    sigma = tensors.pauli(dev)

    bell = solve.bell_circuit(dev)
    _, s, _ = schmidt(bell)
    print(f"Bell state psi_ab = {bell.real.flatten().tolist()}")
    print(f"Schmidt coefficients {s.tolist()}   entropy {entanglement_entropy(bell):.6f}  (ln 2 = 0.693147)")
    print(f"reduced state of qubit 1:\n{reduced_density_1(bell).real}")

    psi0 = bell if params["initial"] == "bell" else tensors.basis_state(0, 0, dev)
    H = tensors.ising_hamiltonian(params["J"], params["h"], sigma)
    result = solve.solve(psi0, H, sigma, params["t_max"])
    S_max, C_max = result["entropy"].max(), result["concurrence"].max()
    print(f"during the evolution: max entropy {S_max:.4f}   max concurrence {C_max:.4f}")
    return plot.animate(result, out_dir / "two_qubit_entanglement.gif")


if __name__ == "__main__":
    cli(run, __file__)
