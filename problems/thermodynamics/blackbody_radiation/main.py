"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    T = tensors.temperatures(params["T_min"], params["T_max"], params["n_temps"], dev)  # (A,)
    lam = tensors.wavelengths(dev)  # (L,)

    r = solve.solve(T, lam)
    exact = equation.stefan_boltzmann(T)  # (A,)
    for a in [0, len(T) // 2, len(T) - 1]:
        print(
            f"T = {T[a]:7.0f} K   λ_max = {1e9 * r['lam_max'][a]:6.1f} nm   λ_max T = {r['lam_max'][a] * T[a]:.4e} m K"
            f"   U = {r['U'][a]:.4e} J/m³  (4σT⁴/c = {exact[a]:.4e})"
        )
    print(f"Wien's constant b = {equation.WIEN_B:.4e} m K")
    return plot.animate(T, lam, r, out_dir / "blackbody_radiation.gif")


if __name__ == "__main__":
    cli(run, __file__)
