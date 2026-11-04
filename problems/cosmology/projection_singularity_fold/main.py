"""Main loop: parameters -> tensors -> solve -> plot. Step 4 of a speculative hypothesis: singularities as folds."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve

RS = 1.0  # lengths in units of the Schwarzschild radius r_s


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator().manual_seed(1959)
    R, a = float(params["fold_radius"]) * RS, float(params["sheet_amplitude"])
    D, samples = int(params["D"]), int(params["samples"])
    if D < 4:
        raise ValueError("need D >= 4: the 3-sphere sheet lives in at least four flat dimensions")

    # 1. known geometry: curved space(time) as a surface in flat space
    em = solve.embeddings(RS, dev)
    print("Known embeddings (induced metric g_ij = eta_AB dz^A/dx^i dz^B/dx^j by autograd, max relative error)")
    for name, err in em["errors"].items():
        print(f"    {name:32s} {err:.2e}")
    print(f"    smallest singular value of dz/dx: Schwarzschild coordinates {em['sv_schw'].min():.1e} at the horizon "
          f"(coordinate artefact), Kruskal coordinates >= {em['sv_krus_min']:.3f}: no fold at r = r_s")  # fmt: skip
    print(f"    z2(r) -> -2 r_s^(3/2) r^(-1/2): z2({em['r_line'][0]:.2f} r_s) = {em['z2'][0]:.2f} r_s; the embedding "
          "runs off to infinity at r = 0 instead of folding")  # fmt: skip

    # 2. GR curvature
    cu = solve.curvature(RS, R, dev)
    print(f"GR  Kretschmann R_abcd R^abcd by autograd vs 12 r_s^2/r^6: max relative error {cu['K_err']:.1e} "
          f"(r from {cu['r'][0]:.2f} to {cu['r'][-1]:.1f} r_s)")  # fmt: skip
    t0 = cu["tidal"][0]
    print(f"    static tidal eigenvalues at r = {cu['r_ext'][0]:.3f} r_s: {t0[0]:+.4f}, {t0[1]:+.4f}, {t0[2]:+.4f} "
          f"(r_s/r^3 = {RS / cu['r_ext'][0] ** 3:.4f}: stretch, squeeze, squeeze)")  # fmt: skip

    # 3. P4: folds of a projection
    fo = solve.folds(a, D, samples, plot.N_FRAMES, gen, dev)
    print("P4  catastrophe exponents of a projected sheet, density ~ |x|^alpha:")
    for n, al in zip(fo["n"].tolist(), fo["alpha_cat"].tolist()):
        print(f"    x = s^{n}  ({plot.CAT_NAMES[n]:11s}) alpha = {al:+.4f}  (exact 1/{n} - 1 = {1 / n - 1:+.4f})")
    print(f"    sphere S^k in R^{D} projected to R^3, rim density ~ u^alpha (u = 1 - |Px|^2):")
    for k, al in zip(fo["k"].tolist(), fo["alpha_sph"].tolist()):
        print(f"      k = {k}: alpha = {al:+.3f}  (exact (k - d - 1)/2 = {(k - 4) / 2:+.3f})")
    print(f"    the wavy sheet (amplitude {a}) folds once tilted past atan(1/a) = {fo['alpha_c']:.3f} rad")

    toy = cu["toy"]
    print(f"Fold shell of radius R = {R:.2f} r_s carrying M = r_s/2 (weak-field caricature, G = c = 1):")
    print(f"    outside R: tidal = r_s/r^3 exactly (Birkhoff / shell theorem); max deviation from GR "
          f"{(toy['tidal_radial'][cu['r'] > R] - RS / cu['r'][cu['r'] > R] ** 3).abs().max():.1e}")  # fmt: skip
    print(f"    at the fold: density ~ (R - r)^{cu['slope_density']:+.3f}, radial tidal ~ (R - r)^"
          f"{cu['slope_tidal']:+.3f}; at the centre tidal = {cu['tidal_centre']:+.3f} (finite)")  # fmt: skip
    print("    GR: Kretschmann ~ r^-6 and tidal ~ r^-3 at r = 0; fold: integrable (R - r)^-1/2 at a shell")
    print(
        "Verdict: the exterior is unchanged (Birkhoff), so any fold inside r_s is invisible from outside; "
        "the interior is not observable directly. Testable versions need the fold outside r_s (step 6)."
    )

    return plot.animate(em, cu, fo, R, a, D, out_dir / "projection_singularity_fold.gif")


if __name__ == "__main__":
    cli(run, __file__)
