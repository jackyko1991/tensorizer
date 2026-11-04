"""Main loop: FIRAS data → batched fit of (T, μ, y) → limits and derived densities → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors

PUBLISHED = {"mu": 9e-5, "y": 1.5e-5}  # Fixsen et al. 1996, 95 % confidence


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    data = tensors.load_firas(dev)
    nu, intensity, sigma = data["nu"], data["intensity"], data["sigma"]
    fit = solve.fit(nu, intensity, sigma, T_init=params["T_init"], steps=params["steps"])

    print(f"FIRAS monopole: {len(nu)} frequencies from {nu[0]:.2f} to {nu[-1]:.2f} cm⁻¹ (Fixsen et al. 1996)")
    for b, name in enumerate(tensors.MODELS):
        T, mu, y = fit["params"][b].tolist()
        sT, smu, sy = fit["sigma"][b].tolist()
        line = f"{name:10s} T = {T:.6f} ± {sT:.6f} K"
        line += f"   μ = {mu:+.2e} ± {smu:.1e}" if smu > 0 else ""
        line += f"   y = {y:+.2e} ± {sy:.1e}" if sy > 0 else ""
        print(f"{line}   χ² = {fit['chi2'][b]:.1f} for {int(fit['dof'][b])} dof")
    mu_lim, y_lim = solve.upper_limits(fit)
    print(
        f"95 % limits: |μ| < {mu_lim:.1e} (published {PUBLISHED['mu']:.0e}),"
        f"  |y| < {y_lim:.1e} (published {PUBLISHED['y']:.1e})"
    )
    print("(ours use only the diagonal errors; the published limits include correlated errors and systematics)")
    T = fit["params"][0, 0]
    n_gamma, u_gamma = equation.photon_number_density(T).item(), equation.energy_density(T).item()
    print(f"from T: photon number density n_γ = {n_gamma:.1f} cm⁻³, energy density u_γ = {u_gamma:.4f} eV cm⁻³")

    # what the plot needs: the T-only model at each fit step until it stops moving
    chi2_t = fit["chi2_history"][:, 0]
    moving = (chi2_t[:-1] - chi2_t[1:]).abs() > 1e-9 * chi2_t[1:]
    n_show = int(moving.nonzero().max()) + 2 if moving.any() else 1
    hist = fit["history"][:n_show, 0]  # (K, 3)
    nu_fine = torch.linspace(0.5, 23.0, 300, dtype=nu.dtype, device=dev)
    T_best, mu_best, y_best = fit["params"][0, 0].item(), 0.0, 0.0
    view = {
        "nu": nu, "data": intensity, "sigma": sigma, "nu_fine": nu_fine,
        "curves": equation.spectrum(nu_fine, hist), "model_at_data": equation.spectrum(nu, hist),
        "T": hist[:, 0], "chi2": chi2_t[:n_show], "dof": fit["dof"][0].item(),
        "mu_limit": solve.distortion_after_refit(nu, sigma, T_best, mu_best - mu_lim, y_best),
        "y_limit": solve.distortion_after_refit(nu, sigma, T_best, mu_best, y_best + y_lim),
    }  # fmt: skip
    view = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in view.items()}
    view["mu_lim"], view["y_lim"] = -mu_lim, y_lim
    view["summary"] = (
        f"T = {T_best:.5f} K (stat. ±{fit['sigma'][0, 0]:.0e})\n"
        f"χ²/dof = {fit['chi2'][0]:.1f}/{int(fit['dof'][0])}\n"
        f"|μ| < {mu_lim:.0e}  (Fixsen: 9e-05)\n|y| < {y_lim:.0e}  (Fixsen: 1.5e-05)\n"
        f"n_γ = {n_gamma:.0f} cm⁻³, u_γ = {u_gamma:.3f} eV/cm³"
    )
    return plot.animate(view, out_dir / "cmb_blackbody.gif")


if __name__ == "__main__":
    cli(run, __file__)
