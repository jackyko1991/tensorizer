"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import binding_energy, per_nucleon, stability_line


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    coeff = tensors.coefficients(params)
    z, n = tensors.grid(params["z_max"], int(1.6 * params["z_max"]), dev)  # (Z, 1), (1, N)

    b = binding_energy(z, n, coeff)  # (Z, N) in MeV
    b_per_a = per_nucleon(b, z, n)  # (Z, N)
    mass, z_best, best = solve.most_stable(b, z, n)  # (K,) each
    z_line = stability_line(mass, coeff)  # (K,)

    a_peak, ba_peak = solve.peak(mass, best)
    print(f"B/A is largest at A = {a_peak:.0f}: {ba_peak:.3f} MeV per nucleon")
    print("nucleus    B formula   B measured   (MeV)")
    for name, zz, a, ba in tensors.MEASURED:
        print(f"{name:7s}  {b[zz - 1, a - zz].item():10.1f}  {ba * a:11.1f}")
    return plot.figure(z, n, b_per_a, mass, best, z_best, z_line, tensors.MEASURED, out_dir / "binding_energy.png")


if __name__ == "__main__":
    cli(run, __file__)
