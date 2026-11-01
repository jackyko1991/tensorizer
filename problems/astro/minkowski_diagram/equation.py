r"""Physics → tensor translation.

Special relativity in one space dimension, with c = 1 (years and light-years). An event is a 4-vector
(here a 2-vector) x^μ = (t, x), μ ∈ {0, 1}. The Minkowski metric with signature (−, +) is

    \eta_{\mu\nu} = \operatorname{diag}(-1, +1)

and the squared interval between two events, the same for every inertial observer, is

    \Delta s^2 = \eta_{\mu\nu}\,\Delta x^\mu \Delta x^\nu = -\Delta t^2 + \Delta x^2      (μ and ν summed)

A boost to a frame moving with velocity β along x is the matrix

    \Lambda^\mu{}_\nu(\beta) = \begin{pmatrix} \gamma & -\gamma\beta \\ -\gamma\beta & \gamma \end{pmatrix},
    \qquad x'^\mu = \Lambda^\mu{}_\nu\, x^\nu,       γ = 1/\sqrt{1 - β^2}

Two boosts in a row are again a boost, Λ(β_1)Λ(β_2) = Λ((β_1 + β_2)/(1 + β_1β_2)) — relativistic velocity
addition. A clock carried along a world line ticks its proper time; for each short straight piece

    d\tau = \sqrt{-\eta_{\mu\nu}\,dx^\mu dx^\nu} = \sqrt{1 - v^2}\,dt .
"""

import torch


def metric(dtype=torch.float64, device=None) -> torch.Tensor:
    r"""η_{μν} = diag(−1, +1): shape (2, 2)."""
    return torch.diag(torch.tensor([-1.0, 1.0], dtype=dtype, device=device))


def lorentz_factor(beta: torch.Tensor) -> torch.Tensor:
    return 1 / torch.sqrt(1 - beta**2)


def boost(beta: torch.Tensor) -> torch.Tensor:
    r"""Λ^μ_ν(β) for any batch of speeds: (...,) → (..., 2, 2)."""
    beta = torch.as_tensor(beta, dtype=torch.float64)
    g = lorentz_factor(beta)
    row0 = torch.stack([g, -g * beta], dim=-1)
    row1 = torch.stack([-g * beta, g], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def transform(Lam: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""x'^μ = Λ^μ_ν x^ν for any batch of events (ν summed).   Lam: (2, 2), x: (..., 2) → (..., 2)"""
    return torch.einsum("mn,...n->...m", Lam, x)


def interval(dx: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    r"""Δs² = η_{μν} Δx^μ Δx^ν (μ, ν summed).   dx: (..., 2) → (...)"""
    return torch.einsum("mn,...m,...n->...", eta, dx, dx)


def proper_time(events: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    r"""τ along a world line given as consecutive events (N, 2): τ_k = Σ_{j<k} \sqrt{-Δs_j^2}, shape (N,)."""
    ds2 = interval(events.diff(dim=0), eta)  # (N-1,) negative for time-like steps
    dtau = torch.sqrt(torch.clamp(-ds2, min=0.0))
    return torch.cat([torch.zeros_like(dtau[:1]), dtau.cumsum(0)])


def velocity_addition(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """(β1 + β2) / (1 + β1 β2): the speed of the combined boost."""
    return (b1 + b2) / (1 + b1 * b2)
