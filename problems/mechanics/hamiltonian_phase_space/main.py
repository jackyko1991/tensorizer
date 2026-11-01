"""Main loop: parameters → Hamiltonian → initial phase-space points → three integrators → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    m, L, g = params["m"], params["length"], 9.81
    H = equation.pendulum(m, L, g) if params["system"] == "pendulum" else equation.harmonic(m, L, g)
    p_sep = 2 * m * L * math.sqrt(g * L)  # momentum at q = 0 on the separatrix H = mgL

    z0 = {
        "orbits": tensors.orbit_fan(params["n_orbits"], p_sep, dev),  # (B, 2) phase portrait
        "test": tensors.test_orbit(params["amplitude"], dev),  # (1, 2) energy comparison
        "blob": tensors.blob((0.0, 0.6 * p_sep), (0.3, 0.1 * p_sep), device=dev),  # (K, 2) Liouville blob
    }
    sol = solve.solve(H, z0, params["dt"], params["t_max"])

    scale = m * g * L  # energy unit: the separatrix sits at H = mgL
    energy = {k: H(tr["test"][:, 0]) for k, tr in sol["traj"].items()}  # (T,) per integrator
    area = {k: equation.shoelace_area(tr["blob"]) for k, tr in sol["traj"].items()}  # (T,) per integrator

    print(f"separatrix energy mgL = {scale:.3f} J, momentum at the bottom 2mL√(gL) = {p_sep:.3f} kg·m²/s")
    period0 = 2 * math.pi * math.sqrt(L / g)
    measured = solve.small_oscillation_period(H, math.radians(2.0), 4.2 * period0, device=dev)
    print(f"small-oscillation period (2° swing): {measured:.4f} s   2π√(L/g) = {period0:.4f} s")
    one_second = int(round(1.0 / params["dt"]))
    for k in solve.STEPPERS:
        err = ((energy[k] - energy[k][0]) / scale).abs()
        ratio = area[k] / area[k][0]
        end = f"{ratio[-1]:.5f}" if k != "Euler" else "too stretched to measure"
        print(
            f"{k:8s}  max |ΔE|/mgL = {err.max():.2e}  (last {err[-1]:.2e})   "
            f"blob area A/A0: after 1 s {ratio[one_second]:.4f}, at the end {end}"
        )
    return plot.animate(sol, energy, area, scale, p_sep, params["system"], out_dir / "hamiltonian_phase_space.gif")


if __name__ == "__main__":
    cli(run, __file__)
