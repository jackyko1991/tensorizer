"""Fit every model of the batch at once: Levenberg–Marquardt steps with the Jacobian from autograd.

Each step linearises I_{bf}(θ_b) around the current θ_b. The Jacobian J_{bfk} = ∂I_{bf}/∂θ_{bk} comes from one
autograd call: the rows of the batch are independent, so differentiating Σ_b I_{bf} with respect to θ (B, 3)
gives every model's own Jacobian. A 3 × 3 solve per model (batched) then gives the step.
"""

import torch

from . import equation, tensors


def jacobian(nu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """J_{bfk} = ∂I_{bf}/∂θ_{bk}: (B, F, 3) by autograd."""
    J = torch.autograd.functional.jacobian(lambda th: equation.spectrum(nu, th).sum(0), theta)  # (F, B, 3)
    return J.permute(1, 0, 2)


def fit(nu, data, sigma, T_init: float = 2.5, steps: int = 40, free: torch.Tensor | None = None) -> dict:
    """data (F,) or (B, F). Returns the fitted θ (B, 3), 1σ errors, χ², degrees of freedom and the history."""
    free = (tensors.FREE if free is None else free).to(nu.device)  # (B, 3) 1 = fitted, 0 = held at 0
    theta = tensors.initial_params(T_init, nu.device)
    lam = torch.full((len(theta),), 1e2, dtype=theta.dtype, device=nu.device)  # damping per model
    eye = torch.eye(3, dtype=theta.dtype, device=nu.device)
    mask = torch.einsum("bi,bj->bij", free, free)  # (B, 3, 3) couplings between fitted parameters only
    pinned = torch.diag_embed(1 - free)  # puts 1 on the diagonal of a held parameter so the solve stays regular

    def normal_equations(th):
        A = jacobian(nu, th) / sigma[:, None]  # (B, F, 3) whitened Jacobian
        r = (data - equation.spectrum(nu, th)) / sigma  # (B, F) whitened residual
        fisher = torch.einsum("bfi,bfj->bij", A, A) * mask + pinned  # repeated f summed: AᵀA
        grad = torch.einsum("bfi,bf->bi", A, r) * free  # Aᵀr
        return fisher, grad

    chi2 = equation.chi2(equation.spectrum(nu, theta), data, sigma)
    history, chi2_history = [theta.clone()], [chi2.clone()]
    for _ in range(steps):
        fisher, grad = normal_equations(theta)
        damped = fisher + lam[:, None, None] * fisher * eye  # Marquardt: scale-free damping of the diagonal
        step = torch.linalg.solve(damped, grad[..., None])[..., 0]  # (B, 3)
        trial = theta + step * free
        trial[:, 0] = trial[:, 0].clamp(min=0.5)  # a temperature step never goes negative
        chi2_trial = equation.chi2(equation.spectrum(nu, trial), data, sigma)
        better = chi2_trial < chi2  # (B,) accept per model
        theta = torch.where(better[:, None], trial, theta)
        chi2 = torch.where(better, chi2_trial, chi2)
        lam = torch.where(better, lam / 3, lam * 4).clamp(1e-9, 1e9)
        history.append(theta.clone())
        chi2_history.append(chi2.clone())

    fisher, _ = normal_equations(theta)
    cov = torch.linalg.inv(fisher) * mask  # (B, 3, 3) covariance of the fitted parameters
    return {
        "params": theta,
        "sigma": torch.diagonal(cov, dim1=-2, dim2=-1).sqrt(),  # (B, 3) 1σ, zero where held
        "cov": cov,
        "chi2": chi2,
        "dof": nu.shape[0] - free.sum(-1),  # (B,)
        "history": torch.stack(history),  # (steps + 1, B, 3)
        "chi2_history": torch.stack(chi2_history),  # (steps + 1, B)
    }


def upper_limits(fit: dict, z: float = 1.96) -> tuple[float, float]:
    """95 % limits |μ| < |μ̂| + 1.96 σ_μ (T + μ model) and |y| < |ŷ| + 1.96 σ_y (T + y model)."""
    p, s = fit["params"], fit["sigma"]
    return (p[1, 1].abs() + z * s[1, 1]).item(), (p[2, 2].abs() + z * s[2, 2]).item()


def distortion_after_refit(nu, sigma, T: float, mu: float, y: float) -> torch.Tensor:
    """What a (μ, y) distortion leaves in the data once T is re-fitted: ΔI minus its best (weighted) T-shift, (F,)."""
    base = torch.tensor([[T, 0.0, 0.0]], dtype=nu.dtype, device=nu.device)
    bent = torch.tensor([[T, mu, y]], dtype=nu.dtype, device=nu.device)
    delta = (equation.spectrum(nu, bent) - equation.spectrum(nu, base))[0]  # (F,)
    dB_dT = jacobian(nu, base)[0, :, 0]  # (F,) the shape of a pure temperature change
    w = sigma**-2
    shift = torch.einsum("f,f,f->", w, dB_dT, delta) / torch.einsum("f,f,f->", w, dB_dT, dB_dT)
    return delta - shift * dB_dT
