"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    k1, k2, length = params["k1"], params["k2"], tensors.PATH_LENGTH  # length = path length l (cm)
    lam = tensors.wavelengths(dev)  # (W,) nm
    E = tensors.absorptivities(lam)  # (W, S) L/(mol cm)
    c0 = tensors.initial_concentrations(params["c0"], dev)  # (S,) mol/L
    t = tensors.sample_times(params["t_end"], dev)  # (T,) s
    noise = tensors.noise((len(t), len(lam)), params["noise"], dev)  # (T, W)

    # 1) the experiment: kinetics → spectra (+ detector noise)
    c, D = solve.simulate(E, c0, t, k1, k2, length, noise)  # (T, S), (T, W)
    # 2) the analysis: spectra → concentrations by least squares
    c_hat, residual, cond = solve.unmix(E, D, length)

    uM = 1e6
    print(f"A → B → C   k1 = {k1:g} s⁻¹, k2 = {k2:g} s⁻¹   [A]0 = {params['c0']:g} µM, l = {length:g} cm")
    print(f"spectra: {len(t)} times × {len(lam)} wavelengths, noise σ = {params['noise']:g}   cond(E) = {cond:.2f}")
    t_peak = math.log(k1 / k2) / (k1 - k2) if (k2 > 0 and k1 != k2) else params["t_end"]
    print(f"B peaks at t ≈ {t_peak:.0f} s with {c[:, 1].max().item() * uM:.2f} µM")
    err = (c_hat - c).abs().max(0).values * uM
    print("max |ĉ − c| per species (µM): " + ", ".join(f"{s} {e:.3f}" for s, e in zip(tensors.SPECIES, err.tolist())))
    print(f"rms residual absorbance = {residual.pow(2).mean().sqrt().item():.4f}  (noise σ = {params['noise']:g})")
    total, total_hat = c.sum(1) * uM, c_hat.sum(1) * uM
    print(f"mass balance A+B+C: true {total.min().item():.6f}–{total.max().item():.6f} µM, "
          f"recovered {total_hat.min().item():.3f}–{total_hat.max().item():.3f} µM")  # fmt: skip

    # isosbestic points: where two spectra cross, the absorbance stays put while that step runs
    iso_ab = equation.crossings(lam, E[:, 0], E[:, 1])
    iso_bc = equation.crossings(lam, E[:, 1], E[:, 2])
    clean = equation.absorbance(E, c, length)  # (T, W) noise-free spectra
    early = t <= t_peak  # mostly A → B
    late = t >= 5.0 / k1  # A is gone (< 1 %): only B → C
    print("isosbestic points, change of absorbance ΔD in each phase (noise-free):")
    for name, points, phase in (("ε_A = ε_B", iso_ab, early), ("ε_B = ε_C", iso_bc, late)):
        if not phase.any():
            continue
        window = clean[phase]  # (T', W)
        swing = window.max(0).values - window.min(0).values  # (W,)
        span = f"t ≤ {t_peak:.0f} s" if phase is early else f"t ≥ {5.0 / k1:.0f} s"
        for x in points:
            w = int((lam - x).abs().argmin())
            print(f"  {name} at λ ≈ {x:.1f} nm, {span}:  ΔD = {swing[w].item():.4f}", end="")
            print(f"   (largest ΔD at any λ: {swing.max().item():.4f})")

    return plot.animate(lam, E, t, c, c_hat, D, iso_ab, iso_bc, out_dir / "beer_lambert_kinetics.gif")


if __name__ == "__main__":
    cli(run, __file__)
