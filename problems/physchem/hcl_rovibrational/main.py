"""Main loop: constants → P and R lines → Boltzmann intensities and spectra for a temperature ramp → fit → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors

N_FRAMES = 60


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    names = tensors.isotope_names()
    mu = tensors.reduced_masses(dev)  # (I,) u
    c = tensors.constants(mu)  # dict of (I,) cm⁻¹
    J = tensors.rotational_levels(params["J_max"], dev)  # (J,)

    R, P = equation.branches(J, c)  # (I, J) cm⁻¹ each
    B0 = equation.rot_constant(0, c["Be"], c["ae"])
    T = tensors.temperatures(params["T"], N_FRAMES, dev)  # (F,) K, the last one is T
    pop = equation.populations(J, B0, c["De"], T)  # (F, I, J)
    I_R, I_P = equation.intensities(J, pop, tensors.abundances(dev))  # (F, I, J) each

    step = min(0.1, params["fwhm"] / 8)
    grid = tensors.wavenumber_grid(P[:, 1:].min().item() - 25, R.max().item() + 25, step, dev)  # (G,)
    S = solve.spectra(R, P, I_R, I_P, grid, params["fwhm"])  # (F, G)

    # "measure" the lines (with a little noise) and fit the constants back by combination differences
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = lambda x: params["noise"] * torch.randn(x.shape, generator=gen).to(dev)  # noqa: E731
    fit = solve.fit_constants(R + noise(R), P + noise(P), J)

    nu0 = equation.band_origin(c["we"], c["wexe"])
    print(f"T = {params['T']:.0f} K   ·   J ≤ {params['J_max']}   ·   line width {params['fwhm']:g} cm⁻¹")
    for i, name in enumerate(names):
        print(
            f"{name}: μ = {mu[i].item():.4f} u   ν₀ = {nu0[i].item():.2f}   B₀ = {B0[i].item():.3f}   "
            f"B₁ = {equation.rot_constant(1, c['Be'], c['ae'])[i].item():.3f} cm⁻¹   "
            f"Morse depth {equation.morse_depth(c['we'], c['wexe'])[i].item():.0f} cm⁻¹"
        )
    print("lines of H³⁵Cl (cm⁻¹):  " + "  ".join(f"R({j})={R[0, j].item():.1f}" for j in range(4)))
    print("                        " + "  ".join(f"P({j})={P[0, j].item():.1f}" for j in range(1, 5)))
    print(
        f"band gap R(0) − P(1) = {(R[0, 0] - P[0, 1]).item():.1f} cm⁻¹ ≈ 2(B₀ + B₁), "
        f"no line at ν₀ (ΔJ = 0 is forbidden)"
    )
    print(
        f"R spacing shrinks {(R[0, 1] - R[0, 0]).item():.2f} → {(R[0, -1] - R[0, -2]).item():.2f}, "
        f"P spacing grows {(P[0, 1] - P[0, 2]).item():.2f} → {(P[0, -2] - P[0, -1]).item():.2f} cm⁻¹  (B₁ < B₀)"
    )
    iso = R[0] - R[1]
    j8 = min(8, len(J) - 1)
    print(
        f"isotope splitting ν(H³⁵Cl) − ν(H³⁷Cl): ν₀ {(nu0[0] - nu0[1]).item():.2f}, "
        f"R(0) {iso[0].item():.2f}, R({j8}) {iso[j8].item():.2f} cm⁻¹"
    )
    J_peak = int(pop[-1, 0].argmax())
    J_estimate = (params["T"] / (2 * equation.HC_K * B0[0].item())) ** 0.5 - 0.5
    print(f"most populated level at {params['T']:.0f} K: J = {J_peak}  (≈ √(kT/2hcB) − ½ = {J_estimate:.1f})")
    print(f"fit (H³⁵Cl, noise {params['noise']:g} cm⁻¹):        true      fitted")
    for key, label in (("Be", "B_e"), ("ae", "α_e"), ("nu0", "ν₀")):
        true = (nu0 if key == "nu0" else c[key])[0].item()
        print(f"   {label:<4} {true:12.4f} {fit[key][0].item():12.4f}")
    print(f"   D    {c['De'][0].item():12.2e} {fit['D'][0].item():12.2e}")

    return plot.animate(grid, S, T, R, P, I_R, J, pop, fit, nu0, names, out_dir / "hcl_rovibrational.gif")


if __name__ == "__main__":
    cli(run, __file__)
