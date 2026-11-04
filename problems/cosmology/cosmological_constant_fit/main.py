"""Main loop: supernova table → batched χ² grid of universes → Adam fit → significance of acceleration."""

from pathlib import Path

from problems._common.runtime import cli, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    # CPU and float64 throughout: the table is small and χ² differences of a few units matter.
    fast = fast_mode()
    z, mu, sigma = tensors.load_distance_moduli(tensors.DATA, z_min=params["z_min"])
    print(f"{len(z)} Type Ia supernovae from {tensors.DATA.name}, z = {z.min():.3f} … {z.max():.2f}")

    n = 31 if fast else params["grid"]
    Om, OL = tensors.parameter_grid(n, n, (0.0, 1.2), (-0.4, 1.6))
    chi2, H0_grid = solve.chi2_grid(z, mu, sigma, Om, OL)
    sig = solve.significance(chi2, Om, OL)
    print(
        f"χ² grid of {n}×{n} universes (H0 solved analytically in each): min χ² = {sig['chi2_min']:.1f} for "
        f"{len(z)} SNe at Ω_m = {sig['best_Om']:.2f}, Ω_Λ = {sig['best_OL']:.2f}, q0 = {sig['best_q0']:+.2f}"
    )
    print(
        f"Ω_Λ ≤ 0 rejected at {sig['sigma_lambda']:.1f}σ, a decelerating universe (q0 ≥ 0) at {sig['sigma_accel']:.1f}σ"
    )
    print(f"share of the likelihood with q0 < 0 (accelerating): {sig['p_accel']:.8f}")

    flat = params["geometry"] == "flat"
    steps = 40 if fast else params["steps"]
    fit = solve.fit(z, mu, sigma, (70.0, params["start_Om"], params["start_OL"]), steps, params["lr"], flat)
    (H0, om, ol), err = fit["best"].tolist(), fit["errors"].tolist()
    print(
        f"Adam ({steps} steps) + L-BFGS, {'flat' if flat else 'free curvature'}: "
        f"χ² {fit['chi2'][0]:.0f} → {fit['chi2'][-1]:.1f}"
    )
    print(f"H0 = {H0:.2f} ± {err[0]:.2f} km/s/Mpc   (set by the SH0ES Cepheid calibration of μ, not by the SNe alone)")
    print(f"Ω_m = {om:.3f} ± {err[1]:.3f}   Ω_Λ = {ol:.3f} ± {err[2]:.3f}   q0 = {equation.deceleration(om, ol):+.3f}")
    print(
        f"χ²/dof = {fit['chi2'][-1]:.0f}/{len(z) - (2 if flat else 3)}: below 1, the diagonal errors are conservative"
    )
    return plot.animate(z, mu, sigma, fit, Om, OL, chi2, sig, out_dir / "cosmological_constant_fit.gif")


if __name__ == "__main__":
    cli(run, __file__)
