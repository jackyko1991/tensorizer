"""Main loop: parameters -> tensors -> solve -> plot."""

import math
from pathlib import Path

import torch

from problems._common.plotting import frame_indices
from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    ns = [1, 2, 3, 4, 5, 6]
    print(
        f"G_F / G_N = G_F M_Pl^2 = {equation.weak_over_gravity():.2e}   "
        f"(M_Pl / v = {tensors.M_PL_GEV / tensors.V_HIGGS:.2e})"
    )

    # (b) gravity on R^3 x T^n: crossover, Gauss's law and the Yukawa fit, for every n at once
    r = torch.logspace(-2, math.log10(30.0), 160, dtype=torch.float64, device=dev)
    cr = solve.crossover(ns, params["m_max"], r)
    print(
        f"image sum vs Kaluza-Klein sum (Poisson summation), 0.6 R < r < 3 R: max relative gap {cr['poisson_gap']:.1e}"
    )
    print("  n   slope at r=0.01R   slope at r=30R   flux small / flux large   Yukawa fit alpha (2n)   lambda/R")
    for i, n in enumerate(ns):
        c = cr["counts"][n]
        tot = (n + 1) * equation.sphere_area(n + 2)
        rs, rl = (
            torch.tensor([1e-3], device=dev, dtype=torch.float64),
            torch.tensor([25.0], device=dev, dtype=torch.float64),
        )
        f_s = (
            equation.sphere_area(n + 2)
            * rs ** (n + 2)
            * equation.radial_field(lambda x, n=n, c=c: equation.potential_images(x, n, c), rs)
        )
        f_l = (
            4
            * math.pi
            * rl**2
            * (2 * math.pi) ** n
            * equation.radial_field(lambda x, n=n, c=c: equation.potential_kk(x, n, c), rl)
        )
        a, lam = cr["fits"][i]
        print(f"  {n}   {cr['slope'][i, 0]:+8.4f}          {cr['slope'][i, -1]:+8.5f}       "
              f"{f_s.item() / tot:.5f} / {f_l.item() / tot:.6f}         "
              f"{a:7.4f} ({2 * n})          {lam:.4f}")  # fmt: skip

    # (a) ADD: radius for n = 1..6 and M* = 1..10 TeV, against torsion balances and the LHC
    n_t = torch.arange(1, 7, dtype=torch.float64, device=dev)
    sweep = torch.logspace(0, 1, 12, dtype=torch.float64, device=dev)
    m_star = torch.cat([torch.tensor([params["M_star"]], dtype=torch.float64, device=dev), sweep])
    add = solve.add_table(n_t, m_star)
    print(f"ADD, M_Pl^2 = M*^(n+2) R^n with M* = {params['M_star']:g} TeV (torsion: Lee 2020, alpha >= 1 above "
          f"{add['lam_limits']['Lee et al.'] * 1e6:.1f} um; LHC: ATLAS 2021 monojet)")  # fmt: skip
    for i, n in enumerate(ns):
        R = add["R"][i, 0].item()
        verdict = "excluded (torsion)" if add["torsion"][i, 0] else ""
        if n == 1:
            verdict = "excluded (solar-system orbits)"
        if add["lhc"][i, 0]:
            verdict += (" + " if verdict else "excluded ") + f"(LHC: M_D > {add['lhc_min'][i]:.1f} TeV)"
        print(f"  n = {n}: R = {R:.3e} m   alpha = {2 * n}   {verdict or 'allowed by these two limits'}")

    # (c) Randall-Sundrum
    krc = torch.linspace(9.0, 14.0, 101, dtype=torch.float64, device=dev)
    tev = solve.rs_scan(krc)
    w = equation.rs_warp(torch.tensor([params["krc"]], dtype=torch.float64)).item()
    print(f"Randall-Sundrum: k r_c = {params['krc']:g} -> e^(-pi k r_c) = {w:.2e}, M_Pl(reduced) x warp = "
          f"{tensors.M_PL_REDUCED_GEV * w / 1e3:.2f} TeV")  # fmt: skip

    zoom = solve.window_scales(len(frame_indices(22, 22)))
    return plot.animate(
        r, ns, cr, zoom, add, m_star, krc, tev, params["krc"], dev, out_dir / "hierarchy_extra_dimensions.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
