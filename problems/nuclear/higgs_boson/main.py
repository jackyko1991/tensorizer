"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    lo, hi = params["m_min"], params["m_max"]
    data = tensors.load_histogram(tensors.DATA, lo, hi, dev)
    x = tensors.design(data["centers"], lo, hi, params["degree"])  # (N, K)
    n_sm = data["sm_signal"].sum().item()
    print(f"{int(data['counts'].sum().item())} diphoton events in {lo:g}–{hi:g} GeV (ATLAS Open Data, 10 fb⁻¹, 13 TeV)")

    # 1. the peak width from the ATLAS simulation, then the free-mass fit to the data
    m_sim, sigma = solve.fit_resolution(data["edges"], data["sm_signal"])
    print(f"simulated H → γγ: peak at {m_sim:.2f} GeV, Gaussian width σ = {sigma:.2f} GeV, {n_sm:.0f} events expected")
    fit = solve.fit_free_mass(data["counts"], x, data["edges"], data["widths"], sigma, params["m_start"], n_sm)
    m_pdg, err_pdg = tensors.M_H_PDG
    print(f"fit: m_H = {fit['m_h']:.2f} ± {fit['m_err']:.2f} GeV   (PDG 2024: {m_pdg} ± {err_pdg})")
    print(f"     signal strength μ = σ/σ_SM = {fit['mu']:.2f} ± {fit['mu_err']:.2f},  local Z = √q₀ = {fit['z']:.2f}σ")
    shape_sm = equation.signal_shape(data["edges"], torch.tensor(fit["m_h"], dtype=torch.float64), sigma)
    b_fit = data["widths"] * equation.background(fit["theta"], x)
    z_exp = equation.significance(solve.q0_known_background(b_fit + n_sm * shape_sm, b_fit, n_sm * shape_sm))
    print(f"     expected for the Standard Model (Asimov, μ = 1): Z = {z_exp.item():.2f}σ")

    # 2. local significance against mass, and background-only toys for the look-elsewhere effect
    masses = tensors.mass_grid(lo + 5, hi - 10, params["scan_step"], dev)  # (M,)
    q0_obs = solve.q0_scan(data["counts"][None], x, data["edges"], data["widths"], masses, sigma)[0]  # (M,)
    best = int(q0_obs.argmax())
    print(f"scan: largest local excess {math.sqrt(q0_obs[best].item()):.2f}σ at {masses[best].item():.1f} GeV")
    theta_b, _, _ = solve.fisher_fit(data["counts"][None], x, data["widths"])
    toys = tensors.toys(data["widths"] * equation.background(theta_b[0], x), params["n_toys"], seed=7)  # (T, N)
    q0_toys = solve.q0_scan(toys, x, data["edges"], data["widths"], masses, sigma)  # (T, M)
    k125 = int((masses - 125.0).abs().argmin())
    q_obs = q0_obs[k125].item()
    lee = solve.global_p_value(q0_toys, q_obs)
    zero = (q0_toys[:, k125] == 0).double().mean().item()
    above = (q0_toys[:, k125] > 1).double().mean().item()
    wilks = 0.5 * math.erfc(math.sqrt(0.5))
    print(f"toys at 125 GeV: q₀ = 0 in {zero:.0%} (Wilks: 50%), P(q₀ > 1) = {above:.3f} (Wilks: {wilks:.3f})")
    print(
        f"look-elsewhere: local p₀(125 GeV) = {lee['p_local']:.2e}; global p ≈ {lee['p_global_gv']:.2e} "
        f"(Gross–Vitells, {lee['n_up']:.2f} upcrossings of q₀ = 0.5); "
        f"toys beating it anywhere: {lee['p_global_toys']:.1%}"
    )

    # 3. the Mexican hat that gives the Higgs its mass
    mu2, lam = tensors.potential_parameters(fit["m_h"], tensors.V_EW)
    print(
        f"Mexican hat: v = (√2 G_F)^(−1/2) = {tensors.V_EW:.2f} GeV, λ = m_H²/2v² = {lam:.4f}, "
        f"minimum at |φ| = v/√2 = {tensors.V_EW / math.sqrt(2):.1f} GeV, "
        f"m_H = √(2λ) v = {math.sqrt(2 * lam) * tensors.V_EW:.2f} GeV"
    )

    return plot.animate(
        data, x, sigma, n_sm, fit, masses, q0_obs, q0_toys, k125, lee, (mu2, lam), out_dir / "higgs_boson.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
