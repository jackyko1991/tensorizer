"""Main loop: sky map → a_ℓm by least squares → quadrupole and octopole axes → Monte Carlo significance."""

from pathlib import Path

import torch

from problems._common.runtime import cli, fast_mode

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    # CPU, float64: the map is tiny (3072 pixels) and eigh of 3×3 matrices is cheap even for 10⁵ skies.
    fast = fast_mode()
    gen = torch.Generator().manual_seed(params["seed"])
    dirs, T, w = tensors.load_sky_map()
    print(f"{tensors.DATA.name}: {len(T)} pixels, rms {T.std():.1f} μK")

    alm = solve.fit_alm(dirs, T, w, params["lmax_fit"])
    D = solve.power(alm)
    print(
        f"band powers ℓ(ℓ+1)C_ℓ/2π: ℓ=2 {D[2]:.0f} μK², ℓ=3 {D[3]:.0f} μK²  (the Planck ΛCDM fit expects ≈ 1000 each)"
    )
    res = solve.alignment_of(alm)
    for ell, key in ((2, "axis2"), (3, "axis3")):
        lon, lat = tensors.lonlat(res[key])
        print(f"ℓ = {ell} axis: (l, b) = ({lon:.1f}°, {lat:+.1f}°)   planarity t = {res[f't{ell}']:.3f}")
    print(f"angle between the axes: {res['angle']:.2f}°   |n̂2·n̂3| = {res['cos']:.5f}")

    n_sims = 5000 if fast else params["n_skies"]
    cos_sims = solve.monte_carlo(n_sims, gen)
    p = solve.p_value(res["cos"], cos_sims)
    print(f"{n_sims} isotropic Gaussian skies: mean |cos| = {cos_sims.mean():.4f} (uniform: 0.5)")
    print(f"p-value P(|cos| ≥ observed) = {p:.5f}   (exact for isotropy: 1 − |cos| = {1 - res['cos']:.5f})")
    t3_sims = solve.axis_of(tensors.isotropic_coefficients(n_sims, 3, gen), 3)[1]
    print(f"octopole planarity t = {res['t3']:.3f}: exceeded by {solve.p_value(res['t3'], t3_sims):.4f} of the skies")
    print("Caveat: the statistic was chosen after seeing the data (look-elsewhere), so the p-value overstates it.")
    return plot.animate(alm, res, cos_sims, out_dir / "axis_of_evil.gif")


if __name__ == "__main__":
    cli(run, __file__)
