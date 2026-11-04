"""Main loop: cited data -> the four criteria -> verdict table -> animation. A speculative hypothesis, tested here."""

from pathlib import Path

import torch

from problems._common.runtime import cli

from . import plot, solve, tensors


def run(params: dict, out_dir: Path, frames: int = 60) -> Path:
    gen = torch.Generator().manual_seed(2018)
    dims, kappa = int(params["dims"]), float(params["speed_factor"])
    z_max, w = float(params["z_max"]), int(params["winding"])
    if dims < 2:
        raise ValueError("a worldline needs time plus at least one other dimension (dims >= 2)")
    c = tensors.cited()

    near = solve.nearby(c)
    print("Criterion 1  nothing mixed in nearby")
    print(
        f"   BESS-Polar 2012: antihelium/helium < {near['antihelium']:.1e} (95% CL); AMS-02: no published antinucleus"
    )
    print(f"   Coma: Fermi-LAT F(>100 MeV) < {c['coma_flux_ul']:.1e} ph/cm^2/s, {near['photons_100']:.2f} photons "
          f"> 100 MeV per annihilation, <sigma v> = {c['pbar_p_sigma_v']:.0e} cm^3/s -> antiproton fraction "
          f"< {near['f_cluster']:.1e}")  # fmt: skip

    d = torch.logspace(2.3, 4.477, frames, dtype=torch.float64)  # 200 Mpc to 30 Gpc
    dom = solve.domain_sweep(c, kappa, z_max, d)
    print("Criterion 2  no matter/antimatter domains inside our horizon (Cohen, De Rujula & Glashow 1998)")
    print(f"   wall annihilation from z = {z_max:.0f} to 20, inflow at {kappa} x the gas sound speed: flux ~ 1/d")
    for dd in (100.0, 1000.0, 10000.0):
        print(f"   d = {dd / 1e3:5.1f} Gpc: worst predicted/observed = {dom['d_min_mpc'] / dd:.2e}")
    print(f"   minimum domain size d_min = {dom['d_min_mpc'] / 1e3:.1f} Gpc (set at E = {dom['E_worst_mev']:.1f} MeV, "
          f"COMPTEL); horizon radius {c['horizon_gpc']} Gpc")  # fmt: skip

    ft = solve.fine_tuning(c, gen)
    print("Criterion 3  fine-tuning of one random worldline")
    print(f"   N_B = {ft['N_B']:.2e} baryons, N_gamma = {ft['N_gamma']:.2e}; random signs: var(Q) / M = "
          f"{(ft['var'] / ft['M']).tolist()}")  # fmt: skip
    print(f"   P(our patch holds +N_B, no antimatter) ~ 10^({ft['log10_today']:.2e}); "
          f"as an excess among N_gamma pairs ~ 10^({ft['log10_early']:.2e})")  # fmt: skip

    mi = solve.mirror(c, dims, w, gen)
    print(f"Criterion 4  CPT mirror (Boyle, Finn & Turok 2018): worldline in D = {dims}, winding w = {w}")
    print(
        f"   net charge on our slices {sorted(set(mi['q_ours'].tolist()))}, in the mirror (its own arrow) "
        f"{sorted(set(mi['q_mirror'].tolist()))}, total {sorted(set((mi['q_ours'] + mi['q_mirror']).tolist()))}"
    )
    print(f"   (a) lightest nu massless: sum m = {mi['sum_no0']:.4f} eV (normal), {mi['sum_io0']:.4f} eV (inverted); "
          f"m_beta = {mi['mb_no0']:.4f} / {mi['mb_io0']:.4f} eV")  # fmt: skip
    print(f"       bounds: KATRIN m_beta < {c['katrin_2022']} (2022), < {c['katrin_2025']} eV (2025); sum m < "
          f"{c['planck_sum_mnu']} (Planck 2018), < {c['desi_2024_sum_mnu']} (DESI 2024), < {c['desi_dr2_sum_mnu']} eV "
          f"(DESI DR2)")  # fmt: skip
    print(f"   (b) dark matter N1 of {c['bft_dm_mass_gev']:.1e} GeV: not testable with current experiments")
    print(
        f"   (c) r = 0 vs BICEP/Keck r < {c['bicep_keck_r']}: consistent; test needs sigma(r) ~ {c['future_sigma_r']}"
    )

    table = solve.verdicts(c, near, dom, ft, mi)
    print("Verdict table")
    for row in table:
        print(f"   {row['criterion']:<48s} {row['status']:<26s} {row['detail']}")
        print(f"   {'':<48s} falsified by: {row['falsify']}")
    print("Bottom line: P3 survives only as one worldline balanced across a CPT mirror universe; then a")
    print("nonzero lightest neutrino mass or primordial B-modes with r >> 0 would falsify it:")
    print("the first test in the series that can fail.")

    maps = solve.domain_pictures(d, 300, gen)
    return plot.animate(c, dom, maps, mi, dims, out_dir / "projection_antimatter_universe.gif")


if __name__ == "__main__":
    cli(run, __file__)
