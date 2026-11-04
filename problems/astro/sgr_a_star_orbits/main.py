"""Main loop: S2 positions + radial velocities → batched Levenberg–Marquardt from many starts → M, R0 → GR effects."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli

from . import plot, solve, tensors
from .equation import (
    AU,
    GM_SUN,
    C,
    kepler_solve,
    pericentre_redshift_kms,
    period,
    precession_per_orbit,
    semi_major_axis_au,
    state_vectors,
)


def run(params: dict, out_dir: Path) -> Path:
    # CPU, float64: e = 0.88 makes Kepler's equation stiff near pericentre, and χ² values reach 10⁷ at bad starts.
    data = tensors.load_s2()
    print(
        f"S2: {len(data['t_a'])} positions ({data['t_a'][0]:.1f}–{data['t_a'][-1]:.1f}) and "
        f"{len(data['t_v'])} radial velocities ({data['t_v'][0]:.1f}–{data['t_v'][-1]:.1f}), Gillessen et al. 2017"
    )
    e = torch.linspace(0.0, 0.99, 50, dtype=torch.float64)[:, None]
    mean = torch.linspace(-math.pi, math.pi, 1001, dtype=torch.float64)
    big_e = kepler_solve(mean, e)
    print(
        f"Kepler solver: max |E − e sin E − M| = {(big_e - e * torch.sin(big_e) - mean).abs().max():.1e} "
        "on 50 × 1001 cases"
    )

    gen = torch.Generator().manual_seed(params["seed"])
    fit = solve.fit(data, n_starts=params["n_starts"], n_iter=params["n_iter"], gen=gen)
    best, err, chi2 = fit["best"], fit["errors"], fit["chi2"]
    good = int((chi2 < chi2.min() + 10).sum())
    print(
        f"\n{params['n_starts']} starts fitted at once; {good} reach the best χ² (within 10): "
        f"χ²_min = {chi2.min():.1f} for {data['n_res']} data"
    )
    labels = {"a": "a (mas)", "e": "e", "inc": "i (°)", "Omega": "Ω (°)", "omega": "ω (°)", "tp": "t_p (yr)",
              "P": "P (yr)", "M": "M (10⁶ M☉)", "R0": "R0 (kpc)"}  # fmt: skip
    for key, label in labels.items():
        print(f"  {label:12s} {best[key]:10.4f} ± {err[key]:.4f}")
    print(
        "published (17 stars, Gillessen et al. 2017): M = 4.28 ± 0.10 (stat) ± 0.21 (sys) × 10⁶ M☉, "
        "R0 = 8.32 ± 0.07 ± 0.14 kpc; S2 period 16.00 yr"
    )

    p = fit["p"][fit["k"] : fit["k"] + 1]
    a_au = semi_major_axis_au(p).item()
    r_p = a_au * (1 - best["e"])
    r_s = 2 * GM_SUN * best["M"] * 1e6 / C**2 / AU  # Schwarzschild radius in AU
    t = torch.linspace(best["tp"] - 0.05, best["tp"] + 0.05, 2001, dtype=torch.float64)
    _, v = state_vectors(p, t)
    v_max = v.norm(dim=-1).max().item() * AU / (365.25 * 86400) / 1000
    print(
        f"\npericentre {r_p:.0f} AU = {r_p * AU / C / 3600:.1f} light-hours = {r_p / r_s:.0f} Schwarzschild radii, "
        f"speed {v_max:.0f} km/s ({v_max / 2997.92:.2f} % of c)"
    )
    print(f"Kepler's third law: P = 2π√(a³/GM) = {period(p).item():.3f} yr with a = {a_au:.0f} AU")
    grav, doppler = pericentre_redshift_kms(p)
    print(
        f"GR prediction at pericentre: gravitational redshift {grav.item():.0f} km/s + transverse Doppler "
        f"{doppler.item():.0f} km/s = {grav.item() + doppler.item():.0f} km/s; GRAVITY 2018 measured "
        f"f = 0.90 ± 0.09 (stat) ± 0.15 (sys) of the GR value (z ≈ 200 km/s / c)"
    )
    dphi = precession_per_orbit(torch.tensor(best["M"] * 1e6), torch.tensor(a_au), torch.tensor(best["e"]))
    print(
        f"Schwarzschild precession 6πGM/(c²a(1−e²)) = {math.degrees(dphi.item()) * 60:.1f}′ per orbit; "
        f"GRAVITY 2020 measured f_SP = 1.10 ± 0.19 of the GR value (≈ 12′)"
    )
    return plot.animate(data, fit, out_dir / "sgr_a_star_orbits.gif")


if __name__ == "__main__":
    cli(run, __file__)
