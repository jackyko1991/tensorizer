"""Main loop: load data → initial guess → fit (maximum likelihood) → uncertainties → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    t, x, sigma = tensors.load_measurements(dev)  # (N,) each
    N = len(t)
    jitter = params["jitter"] == "on"
    p0 = tensors.initial_parameters(t, x, sigma, params["omega0"], params["gamma0"])  # (6,)

    p, history, _ = solve.fit(p0, t, x, sigma, params["lr"], params["iterations"], jitter)
    chi2_hist = torch.stack([torch.sum(equation.residuals(q, t, x, sigma, jitter) ** 2) for q in history])

    # uncertainties: Hessian of -2 ln L (with s), or Gauss–Newton (JᵀJ)⁻¹ for plain χ²
    if jitter:
        C = equation.covariance_from_hessian(p, t, x, sigma)  # (6, 6)
    else:
        C = equation.covariance_from_jacobian(equation.jacobian(p, t, x, sigma, jitter=False))  # (5, 5)
    err = C.diagonal().sqrt()
    print(f"N = {N} points, {'maximum likelihood with jitter s' if jitter else 'weighted least squares (s = 0)'}")
    for i, name in enumerate(equation.NAMES[: len(err)]):
        print(f"  {name:7s} = {p[i].item():9.4f} ± {err[i].item():.4f}")
    if jitter:
        s = p[5].exp().item()
        print(f"  s       = {s:9.4f} ± {s * err[5].item():.4f} mm   (extra scatter beyond the error bars)")

    # χ²/dof with and without the error term s
    r = equation.residuals(p, t, x, sigma, jitter)
    print(f"χ²/dof using σ_tot = {(r @ r).item() / (N - len(err)):.3f}   (should be ≈ 1)")
    p_ls, _, _ = solve.fit(p0, t, x, sigma, params["lr"], params["iterations"], jitter=False)
    print(
        f"χ²/dof using σ_i only (weighted least squares fit) = {equation.chi2(p_ls, t, x, sigma).item() / (N - 5):.3f}"
    )

    # the linear sub-problem at the fitted γ, ω (and s): lstsq and normal equations give the same
    # β = (A cos φ, A sin φ, c), and the same A, φ, c as the nonlinear fit
    w = equation.total_sigma(sigma, p[5]) if jitter else sigma
    beta = equation.linear_fit(t, x, w, p[1].item(), p[2].item())
    beta_ne = equation.normal_equations(t, x, w, p[1].item(), p[2].item())
    A, phi = equation.amplitude_phase(beta)
    print(f"linear sub-problem at fitted γ, ω: lstsq β = {[round(b, 4) for b in beta.tolist()]}")
    print(
        f"  normal equations β = {[round(b, 4) for b in beta_ne.tolist()]}  →  A = {A.item():.4f}, φ = {phi.item():.4f}"
    )
    return plot.animate(t, x, sigma, history, chi2_hist, jitter, out_dir / "measurement_fitting.gif")


if __name__ == "__main__":
    cli(run, __file__)
