"""Main loop: pulsars + random plane-wave skies → timing residuals → pair correlations → HD vs alternatives."""

from pathlib import Path

import torch

from problems._common.runtime import cli, fast_mode

from . import plot, solve, tensors
from .equation import hd_minimum, hellings_downs, separation_cos


def run(params: dict, out_dir: Path) -> Path:
    # CPU, float64: the pulsar-term phases ω L (1 + Ω̂·p̂)/c are hundreds of radians and need double precision.
    fast = fast_mode()
    gen = torch.Generator().manual_seed(params["seed"])
    n_pulsars = 30 if fast else params["n_pulsars"]
    n_real = 5 if fast else params["n_realisations"]
    n_waves = 200 if fast else params["n_waves"]

    # --- checks of the geometry: antenna-pattern sky average and the HD normalisation ---
    probe = tensors.random_directions(1, gen)[0]
    avg = solve.sky_average_power(probe, 20000 if fast else 200000, gen)
    print(f"sky average of Σ_A (F^A)² for one pulsar: {avg:.4f}  (exact 1/3 = {1 / 3:.4f})")
    zetas = torch.tensor([10.0, 45.0, 82.5, 120.0, 170.0], dtype=torch.float64)
    numeric = solve.hd_from_sky_integral(zetas, 20000 if fast else 200000, gen)
    exact = hellings_downs(torch.cos(torch.deg2rad(zetas)))
    for z, a, b in zip(zetas.tolist(), numeric.tolist(), exact.tolist()):
        print(f"Γ({z:5.1f}°): sky integral (3/8π)∮ΣF_aF_b dΩ = {a:+.4f}   formula = {b:+.4f}")
    z_min, g_min = hd_minimum()
    print(
        f"Γ(0⁺) = {hellings_downs(torch.tensor(1.0)).item():.3f}, Γ(180°) = 0.25, "
        f"minimum Γ = {g_min:.4f} at ζ = {z_min:.1f}°"
    )

    # --- the timing array ---
    pulsars = tensors.random_directions(n_pulsars, gen)  # (N, 3)
    distances = tensors.pulsar_distances(n_pulsars, gen)  # (N,) light-years
    t = tensors.time_grid(params["n_times"], params["span"])  # (T,) years
    print(
        f"\n{n_pulsars} pulsars ({n_pulsars * (n_pulsars - 1) // 2} pairs), {params['span']:.0f} yr, "
        f"{n_real} skies of {n_waves} plane waves, white noise {params['noise']} × GW rms"
    )
    r = solve.simulate(pulsars, distances, t, n_real, n_waves, params["n_freq"], params["noise"], gen)  # (K, N, T)

    cos_zeta = separation_cos(pulsars)
    zeta, rho = solve.pair_correlations(r, cos_zeta)  # (P,), (K, P)
    edges = tensors.bin_edges(params["n_bins"])
    centres = (edges[1:] + edges[:-1]) / 2
    binned = solve.bin_by_angle(zeta, rho, edges)  # (K, B)
    err = solve.bin_errors(binned)

    one_sky = solve.fit_models(centres, binned[0], binned.std(dim=0))
    all_skies = solve.fit_models(centres, binned.mean(0), err)
    for label, fits in (("one sky", one_sky), (f"mean of {n_real} skies", all_skies)):
        text = ",  ".join(f"{name}: A = {a:+.2f}, χ² = {chi:.1f}" for name, (a, chi) in fits.items())
        print(f"{label:16s} ({len(centres)} bins) {text}")
    expected = 1 / (1 + params["noise"] ** 2)
    print(f"HD amplitude {all_skies['Hellings–Downs'][0]:.2f}: noise dilutes it towards 1/(1 + σ²) = {expected:.2f}")
    best = min(all_skies, key=lambda name: all_skies[name][1])
    print(f"best model: {best}")

    a = 0
    sep = torch.rad2deg(torch.acos(cos_zeta[a].clamp(-1, 1)))
    sep[a] = 999.0
    near = int(sep.argmin())
    mid = int((sep - 82.5).abs().argmin())
    return plot.animate(
        pulsars, t, r, zeta, rho, edges, binned, [a, near, mid], out_dir / "pulsar_timing_hellings_downs.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
