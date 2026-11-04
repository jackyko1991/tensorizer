"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    data = tensors.load_sparc(dev)
    data["prior_dex"] = params["prior_dex"] if params["prior_dex"] > 0 else None
    fit = solve.fit_all(data, iterations=params["iterations"], lr=params["lr"])
    R_fine, curves = solve.model_curves(fit, data)

    print(f"SPARC (Lelli, McGaugh & Schombert 2016): {len(data['names'])} galaxies, {int(data['mask'].sum())} points")
    print(f"one batched fit: 4 models × {len(data['names'])} galaxies, Υ prior ±{params['prior_dex']} dex")
    print(f"{'galaxy':10s} " + " ".join(f"{m:>9s}" for m in solve.MODELS) + "   (reduced χ²)")
    for g, name in enumerate(data["names"]):
        print(f"{name:10s} " + " ".join(f"{float(fit[m]['chi2_nu'][g]):9.2f}" for m in solve.MODELS))
    for g, name in enumerate(data["names"]):
        print(f"{name:10s} baryons-only needs Υ_disk = {float(fit['baryons']['ups_d'][g]):5.2f}   "
              f"NFW: Υ_d = {float(fit['nfw']['ups_d'][g]):.2f}, r_s = {float(fit['nfw']['r_s'][g]):6.1f} kpc   "
              f"Burkert: r_0 = {float(fit['burkert']['r_s'][g]):5.1f} kpc")  # fmt: skip
    a0 = float(fit["mond"]["a0"]) * tensors.MS2_PER_KMS2_KPC
    print(f"MOND / RAR: one a₀ for all galaxies = {a0:.3e} m/s² (McGaugh et al. 2016: 1.20e-10)")
    print("Dark matter is inferred from gravity, not directly detected; MOND fits galaxies but struggles with")
    print("clusters and the CMB.")
    return plot.animate(data, fit, R_fine, curves, out_dir / "dark_matter_rotation.gif")


if __name__ == "__main__":
    cli(run, __file__)
