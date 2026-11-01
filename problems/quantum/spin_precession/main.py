"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    sigma = tensors.pauli(dev)
    psi0 = tensors.tilted_spinor(params["theta0"], dev)

    times, psi, s = solve.solve(
        psi0,
        sigma,
        params["gamma"],
        params["B0"],
        params["B1"],
        params["drive_freq"],
        params["t_max"],
    )

    print(f"Larmor frequency gamma*B0 = {params['gamma'] * params['B0']:.3f}")
    print(f"Rabi frequency (resonant, RWA) gamma*B1/2 = {params['gamma'] * params['B1'] / 2:.3f}")
    print(f"min s_z = {s[:, 2].min():.4f}   max | |s| - 1 | = {(s.norm(dim=1) - 1).abs().max():.2e}")
    return plot.animate(times, s, out_dir / "spin_precession.gif")


if __name__ == "__main__":
    cli(run, __file__)
