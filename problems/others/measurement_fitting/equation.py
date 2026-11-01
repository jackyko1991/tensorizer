r"""Physics → tensor translation: a measurement model with an explicit error term, and its likelihood.

Each measured point is the model plus an error:

    x_i = f(t_i; θ) + ε_i,      ε_i ~ N(0, σ_{tot,i}²),      σ_{tot,i}² = σ_i² + s²

    f(t; θ) = A e^{-γt} cos(ωt + φ) + c,       θ = (A, γ, ω, φ, c)

σ_i is the error bar quoted with point i; s is an extra scatter ("jitter") that the error bars missed,
fitted together with θ. All parameters live in one vector p = (A, γ, ω, φ, c, ln s), shape (6,);
s = e^{ln s} stays positive whatever the optimiser does.

Maximum likelihood: minimise

    -2 ln L = Σ_i [ r_i² + ln(2π σ_{tot,i}²) ],     r_i = (x_i - f_i) / σ_{tot,i}     (i summed)

With s = 0 the log term is a constant and -2 ln L = χ² + const, χ² = Σ_i ((x_i - f_i)/σ_i)²:
weighted least squares is maximum likelihood when the error bars are right.

Uncertainties: J_{ip} = ∂r_i/∂θ_p, and C_{pq} = (J_{ip} J_{iq})⁻¹  (i summed: Gauss–Newton), or, including s,
C = 2 H⁻¹ with H_{pq} = ∂²(-2 ln L)/∂p_p ∂p_q (the Hessian).
"""

import math

import torch

NAMES = ["A", "gamma", "omega", "phi", "offset", "log_s"]


def model(t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """f_i = A e^{-γ t_i} cos(ω t_i + φ) + c.   t: (N,), theta: (≥5,) → (N,)."""
    A, gamma, omega, phi, offset = theta[0], theta[1], theta[2], theta[3], theta[4]
    return A * torch.exp(-gamma * t) * torch.cos(omega * t + phi) + offset


def total_sigma(sigma: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
    """σ_{tot,i} = √(σ_i² + s²): quoted error bar and extra scatter add in quadrature.   (N,)"""
    return torch.sqrt(sigma**2 + torch.exp(2 * log_s))


def residuals(p: torch.Tensor, t, x, sigma, jitter: bool = True) -> torch.Tensor:
    """Normalised residuals r_i = (x_i - f_i) / σ_{tot,i} (σ_i alone when jitter is off).   (N,)"""
    s_tot = total_sigma(sigma, p[5]) if jitter else sigma
    return (x - model(t, p)) / s_tot


def neg2_log_likelihood(p: torch.Tensor, t, x, sigma) -> torch.Tensor:
    """-2 ln L = Σ_i [ r_i² + ln(2π σ_{tot,i}²) ]: the misfit plus a penalty for blaming it on a large s."""
    s_tot = total_sigma(sigma, p[5])
    r = (x - model(t, p)) / s_tot
    return torch.sum(r**2 + torch.log(2 * math.pi * s_tot**2))


def chi2(p: torch.Tensor, t, x, sigma) -> torch.Tensor:
    """χ² = Σ_i ((x_i - f_i)/σ_i)²: weighted least squares, i.e. -2 ln L with s = 0 (minus a constant)."""
    r = residuals(p, t, x, sigma, jitter=False)
    return torch.einsum("i,i->", r, r)


def jacobian(p: torch.Tensor, t, x, sigma, jitter: bool = True) -> torch.Tensor:
    """J_{ip} = ∂r_i/∂θ_p for the five model parameters (s held fixed), by autograd.   (N, 5)"""
    rest = p[5:]
    return torch.autograd.functional.jacobian(
        lambda theta: residuals(torch.cat([theta, rest]), t, x, sigma, jitter), p[:5]
    )


def covariance_from_jacobian(J: torch.Tensor) -> torch.Tensor:
    """C_{pq} = (J_{ip} J_{iq})⁻¹, i summed: the Gauss–Newton covariance of θ.   (N, 5) → (5, 5)"""
    return torch.linalg.inv(torch.einsum("ip,iq->pq", J, J))


def covariance_from_hessian(p: torch.Tensor, t, x, sigma) -> torch.Tensor:
    """C = 2 H⁻¹, H = Hessian of -2 ln L in all six parameters (including ln s).   (6, 6)"""
    H = torch.autograd.functional.hessian(lambda q: neg2_log_likelihood(q, t, x, sigma), p)
    return 2 * torch.linalg.inv(H)


# --- the linear sub-problem: for fixed γ, ω the model is linear in (a, b, c) ---------------------------


def design_matrix(t: torch.Tensor, gamma: float, omega: float) -> torch.Tensor:
    """D_{ik} with columns e^{-γt} cos ωt, -e^{-γt} sin ωt, 1, so f_i = D_{ik} β_k with β = (A cos φ, A sin φ, c).

    (cos(ωt + φ) = cos ωt cos φ - sin ωt sin φ.)   t: (N,) → (N, 3)
    """
    e = torch.exp(-gamma * t)
    return torch.stack([e * torch.cos(omega * t), -e * torch.sin(omega * t), torch.ones_like(t)], dim=1)


def linear_fit(t, x, sigma, gamma: float, omega: float) -> torch.Tensor:
    """Weighted linear least squares by torch.linalg.lstsq: rows divided by σ_i.   → β (3,)"""
    D = design_matrix(t, gamma, omega) / sigma[:, None]  # (N, 3)
    return torch.linalg.lstsq(D, (x / sigma)[:, None]).solution[:, 0]


def normal_equations(t, x, sigma, gamma: float, omega: float) -> torch.Tensor:
    """The same β from the normal equations (D_{ik} w_i D_{il}) β_l = D_{ik} w_i x_i, w_i = 1/σ_i².   (3,)"""
    D = design_matrix(t, gamma, omega)
    w = 1 / sigma**2
    M = torch.einsum("ik,i,il->kl", D, w, D)  # (3, 3)
    rhs = torch.einsum("ik,i,i->k", D, w, x)  # (3,)
    return torch.linalg.solve(M, rhs)


def amplitude_phase(beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(a, b) = (A cos φ, A sin φ) → (A, φ)."""
    return torch.hypot(beta[0], beta[1]), torch.atan2(beta[1], beta[0])
