"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import left_probability, two_state_prediction


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    V0, a, n = params["V0"], params["a"], params["n_grid"]
    x, dx = tensors.grid(n, 4 * a + 6, dev)  # box reaches well past both minima
    V = tensors.double_well(x, V0, a)
    H = tensors.hamiltonian(x, dx, V)

    energies, phi = solve.eigenstates(H, x, dx)
    period = solve.tunnelling_period(energies)
    times, psi = solve.solve(phi, energies, tensors.left_state(n, dev), params["periods"] * period)
    p_left = left_probability(psi, x, dx)
    delta_E = (energies[1] - energies[0]).item()

    print(f"E_0 = {energies[0]:.5f}   E_1 = {energies[1]:.5f}   barrier V0 = {V0:.3f}")
    print(f"splitting dE = {delta_E:.5f}   tunnelling period 2pi/dE = {period:.2f}")
    print(f"P_left: min {p_left.min():.4f}  (two-state cos^2 prediction: 0)")
    prediction = two_state_prediction(times, delta_E)
    return plot.animate(x, times, psi, V, phi, energies, p_left, prediction, out_dir / "double_well.gif")


if __name__ == "__main__":
    cli(run, __file__)
