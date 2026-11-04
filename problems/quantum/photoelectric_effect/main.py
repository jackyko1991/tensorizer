"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    phi, S, n_photons = params["phi"], params["irradiance"], params["n_photons"]
    torch.manual_seed(0)

    # Monte Carlo: every frequency of the sweep × 3 intensities × every photon, in one batch
    nu = tensors.frequencies(3.5, 13.5, 91)  # 10^14 Hz
    intensities = torch.tensor([0.5, 1.0, 2.0])
    voltages = tensors.retarding_voltages(-1.0, 4.0)
    sim = solve.simulate(nu, phi, intensities, n_photons, voltages=voltages)

    synthetic = solve.fit_line(nu, sim["V_stop"][1])
    nu0 = equation.threshold_frequency(phi)
    print(f"threshold ν0 = φ/h = {nu0:.3f}×10¹⁴ Hz  (λ0 = {2997.92458 / nu0:.0f} nm)")
    print(
        f"simulated stopping potentials: slope {synthetic['slope']:.5f} V/10¹⁴ Hz (h/e = {equation.H_OVER_E:.5f}),"
        f" intercept {synthetic['intercept']:.3f} V (−φ/e = {-phi:.3f})"
    )
    spread = (sim["V_stop"] - sim["V_stop"][1]).abs().max()
    print(
        f"stopping potential at 0.5×, 1×, 2× intensity differs by at most {spread:.4f} V; saturation currents"
        f" {sim['current'][:, -1, 0].tolist()}"
    )

    # Millikan (1916), sodium
    nu_m, V_m = tensors.millikan_sodium()
    fit = solve.fit_line(nu_m, V_m)
    h = fit["slope"] * 1e-14 * equation.E_CHARGE
    print(f"Millikan 1916 sodium, {len(nu_m)} lines: slope {fit['slope']:.4f} ± {fit['slope_err']:.4f} V/10¹⁴ Hz")
    print(f"  h = {h:.4e} J s  vs today {equation.H_PLANCK:.5e} J s  ({100 * (h / equation.H_PLANCK - 1):+.2f} %)")
    nu0_m = -fit["intercept"] / fit["slope"]
    print(f"  intercept: φ = {-fit['intercept']:.2f} ± {fit['intercept_err']:.2f} eV, ν0 = {nu0_m:.2f}×10¹⁴ Hz")

    area = math.pi * (1e-10) ** 2  # one atom
    t_wave = equation.classical_delay(S, phi, area)
    print(
        f"wave picture at {S:g} W/m²: an electron needs {t_wave:.0f} s ({t_wave / 60:.1f} min) to collect φ;"
        " observed delay < 3 ns (Lawrence & Beams 1928)"
    )
    return plot.animate(nu, sim, phi, intensities, (nu_m, V_m, fit), t_wave, out_dir / "photoelectric_effect.gif")


if __name__ == "__main__":
    cli(run, __file__)
