"""Main loop: thermochemistry → equilibrium at every temperature → adiabatic flame temperature → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

T_MIN = 1000.0  # K, start of the temperature sweep
T_COLD = 1000.0  # K, "undissociated" reference composition for the frozen flame temperature


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    t_start = time.perf_counter()
    P, phi0 = params["P_bar"], params["phi"]
    A = tensors.element_matrix(dev)  # (E, S)
    th = tensors.thermo(dev)  # each (S,)

    # --- 1) one feed, a batch of temperatures: composition x_j(T) ---
    T = tensors.temperatures(T_MIN, params["T_max"], 30 if fast else 111, dev)  # (B,)
    b = tensors.feed(torch.tensor([phi0], dtype=torch.float64, device=dev))[0]  # (E,)
    sweep = solve.temperature_sweep(T, A, b, P, th, n_iter=80)
    n = sweep["n"]  # (B, S)
    g = equation.standard_potential(T, th)
    G = equation.gibbs(n, g, P)  # (B,) G/RT at the minimum
    balance, kkt = solve.residuals(A, b.expand(len(T), -1), n, sweep["pi"], g, P)

    # autograd check: ∂(G/RT)/∂n_j at the minimum equals μ_j/RT = π_i A_ij (element potentials)
    n_req = n.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(equation.gibbs(n_req, g, P).sum(), n_req)
    major = equation.mole_fractions(n) > 1e-8
    grad_err = (grad - torch.einsum("bi,ij->bj", sweep["pi"], A))[major].abs().max().item()

    # --- 2) a batch of feeds (equivalence ratios): adiabatic flame temperature ---
    phis = torch.linspace(0.6, 1.5, 7 if fast else 19, dtype=torch.float64, device=dev)  # (F,)
    if not torch.isclose(phis, torch.tensor(phi0, dtype=torch.float64)).any():
        phis = torch.sort(torch.cat([phis, torch.tensor([phi0], dtype=torch.float64, device=dev)])).values
    bF, H_r = tensors.feed(phis), tensors.reactant_enthalpy(phis)  # (F, E), (F,)
    T_ad, n_ad = solve.flame_temperature(A, bF, H_r, P, th)
    cold = solve.equilibrium(equation.standard_potential(torch.full_like(phis, T_COLD), th), A, bF, P)["n"]
    T_frozen = solve.frozen_flame_temperature(cold, H_r, th)
    k0 = int(torch.argmin((phis - phi0).abs()))

    x = equation.mole_fractions(n)
    names = tensors.SPECIES
    print(f"CH4 + {2 / phi0:.3g} (O2 + 3.76 N2)   ·   φ = {phi0}   ·   P = {P} bar   ·   {len(names)} species, "
          f"{len(tensors.ELEMENTS)} elements, {len(T)} temperatures solved as one batch")  # fmt: skip
    print("mole fractions:   " + "".join(f"{s:>9}" for s in names))
    for T_show in (1500.0, 2000.0, 2500.0, 3000.0, 3500.0):
        if T_show <= params["T_max"]:
            i = int(torch.argmin((T - T_show).abs()))
            print(f"  T = {T[i].item():6.0f} K     " + "".join(f"{v:9.2e}" for v in x[i].tolist()))
    print(f"optimality: max |A n - b| = {balance:.1e}   ·   max |μ_j/RT - π_i A_ij| = {kkt:.1e}   ·   "
          f"autograd ∂(G/RT)/∂n_j vs π_i A_ij: {grad_err:.1e}")  # fmt: skip
    print(f"adiabatic flame temperature at φ = {phis[k0].item():.2f}: {T_ad[k0].item():.0f} K with dissociation, "
          f"{T_frozen[k0].item():.0f} K if the products did not dissociate")  # fmt: skip
    k_max = int(torch.argmax(T_ad))
    print(f"hottest flame over φ ∈ [{phis[0].item():.1f}, {phis[-1].item():.1f}]: "
          f"{T_ad[k_max].item():.0f} K at φ = {phis[k_max].item():.2f}   ·   solved in "
          f"{time.perf_counter() - t_start:.1f} s")  # fmt: skip
    x_ad = equation.mole_fractions(n_ad)[k0]  # (S,) at the flame temperature
    print(f"at T_ad (φ = {phis[k0].item():.2f}): x_CO = {x_ad[names.index('CO')].item():.2e}, "
          f"x_NO = {x_ad[names.index('NO')].item():.2e} (equilibrium, no kinetics)")  # fmt: skip

    return plot.animate(
        T, x, G, phis, T_ad, T_frozen, phis[k0].item(), T_ad[k0].item(), names, out_dir / "gibbs_minimisation.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
