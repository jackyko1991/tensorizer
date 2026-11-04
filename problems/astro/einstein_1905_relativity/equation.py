r"""Physics → tensor translation.

Einstein, "On the electrodynamics of moving bodies" (1905), from two postulates: (1) the laws of physics are the
same in every inertial frame; (2) light moves at c in every inertial frame. Units with c = 1 unless stated.

A light clock (a photon bouncing between mirrors a distance D apart) moving sideways at speed β: the photon runs
along the diagonal at speed 1, so one tick takes

    \Delta t = \gamma\,\Delta\tau,\qquad \gamma = 1/\sqrt{1 - \beta^2}

and the same argument for a clock lying along the motion gives the contraction L = L_0/γ. Events are 4-vectors
x^μ = (t, x, y, z), the metric is η = diag(−1, +1, +1, +1) and a boost along x is

    x'^\mu = \Lambda^\mu{}_\nu(\beta)\,x^\nu,\qquad \Delta s^2 = \eta_{\mu\nu}\,\Delta x^\mu\Delta x^\nu  (ν, μ summed)

Speeds add as w = (u + v)/(1 + uv), never beyond 1. A body's 4-momentum p^μ = (γm, γmβ, 0, 0) satisfies
η_{μν} p^μ p^ν = −m², i.e. E² = (pc)² + (mc²)².

"Does the inertia of a body depend upon its energy content?" (1905): a body at rest emits light energy L/2 in two
opposite directions. Seen from a frame moving at β the two pulses carry (L/2)γ(1 ∓ β cos θ), together γL, so the
body's kinetic energy drops by K_0 − K_1 = L(γ − 1) — the kinetic energy of a mass L/c²: it lost mass Δm = L/c².
A nucleus weighs less than its parts by its binding energy: B = Z m_H + N m_n − M(Z, N), in u × 931.494 MeV.
"""

import torch

U_MEV = 931.49410372  # MeV per atomic mass unit (CODATA 2022)
M_H = 1.007825031898  # u, hydrogen-1 atom (AME2020): proton + electron
M_N = 1.00866491606  # u, neutron


def gamma(beta: torch.Tensor) -> torch.Tensor:
    """Lorentz factor γ = 1/√(1 − β²), any shape."""
    return 1 / torch.sqrt(1 - beta**2)


def metric(dtype=torch.float64) -> torch.Tensor:
    """η_{μν} = diag(−1, +1, +1, +1): (4, 4)."""
    return torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=dtype))


def boost(beta: torch.Tensor) -> torch.Tensor:
    r"""Λ^μ_ν(β) along x for a batch of speeds: (...,) → (..., 4, 4)."""
    beta = torch.as_tensor(beta, dtype=torch.float64)
    g = gamma(beta)
    Lam = torch.eye(4, dtype=torch.float64).expand(*beta.shape, 4, 4).clone()
    Lam[..., 0, 0] = g
    Lam[..., 1, 1] = g
    Lam[..., 0, 1] = -g * beta
    Lam[..., 1, 0] = -g * beta
    return Lam


def transform(Lam: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""x'^μ = Λ^μ_ν x^ν (ν summed).   Lam: (..., 4, 4), x: (K, 4) → (..., K, 4)"""
    return torch.einsum("...mn,kn->...km", Lam, x)


def interval(dx: torch.Tensor) -> torch.Tensor:
    r"""Δs² = η_{μν} Δx^μ Δx^ν (μ, ν summed).   dx: (..., 4) → (...)"""
    return torch.einsum("mn,...m,...n->...", metric(dx.dtype), dx, dx)


def velocity_addition(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """w = (u + v)/(1 + uv): the speed of v seen from a frame moving at −u (broadcasts)."""
    return (u + v) / (1 + u * v)


def four_momentum(m: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """p^μ = (γm, γmβ, 0, 0) in MeV (broadcast m against beta): (..., 4)."""
    g = gamma(beta)
    E, p = torch.broadcast_tensors(g * m, g * m * beta)
    zero = torch.zeros_like(E)
    return torch.stack([E, p, zero, zero], dim=-1)


def pulse_energies(L: float, beta: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Energy of the two light pulses (L/2 each at angles θ and θ + π in the rest frame) seen from a frame moving
    at β (relativistic Doppler factor γ(1 − β cos θ)): (..., 2)."""
    angles = torch.stack([theta, theta + torch.pi], dim=-1)  # (..., 2)
    return 0.5 * L * gamma(beta)[..., None] * (1 - beta[..., None] * torch.cos(angles))


def binding_energy(Z: torch.Tensor, N: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """B = (Z m_H + N m_n − M) c² in MeV from atomic masses in u: (K,)."""
    return (Z * M_H + N * M_N - M) * U_MEV
