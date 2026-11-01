r"""Physics → tensor translation.

Energy and momentum form one four-vector p^μ = (E, p_x, p_y, p_z) (units with c = 1, GeV). With the metric
η_{μν} = diag(1, −1, −1, −1) the invariant "length" is the mass,

    p_μ p^μ = η_{μν} p^μ p^ν = E^2 - \mathbf p^2 = m^2        (μ, ν summed).

A particle of mass M at rest decays into m_1 and m_2 flying back to back with momentum

    p^* = \frac{\sqrt{[M^2 - (m_1 + m_2)^2]\,[M^2 - (m_1 - m_2)^2]}}{2M}.

Moving to the lab, where the parent has velocity \boldsymbol β, every four-vector is transformed by the boost

    p'^μ = Λ^μ{}_ν \, p^ν,     Λ^0{}_0 = γ,   Λ^0{}_i = Λ^i{}_0 = γ β_i,   Λ^i{}_j = δ_{ij} + (γ - 1) β_i β_j / β^2.

The invariant mass of the decay products, (p_1 + p_2)_μ (p_1 + p_2)^μ = M^2, is the same in every frame.
"""

import math

import torch


def metric(device: torch.device) -> torch.Tensor:
    r"""η_{μν} = diag(1, −1, −1, −1): shape (4, 4)."""
    return torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64, device=device))


def minkowski(p: torch.Tensor, q: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    r"""p·q = η_{μν} p^μ q^ν   (μ, ν summed).   p, q: (..., 4) → (...)"""
    return torch.einsum("...m,mn,...n->...", p, eta, q)


def invariant_mass(p: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    r"""m = \sqrt{p_μ p^μ}.   (..., 4) → (...)"""
    return minkowski(p, p, eta).clamp(min=0.0).sqrt()


def rest_frame_momentum(big_m: float, m1: float, m2: float) -> float:
    r"""p^* of the two daughters in the parent's rest frame."""
    return math.sqrt((big_m**2 - (m1 + m2) ** 2) * (big_m**2 - (m1 - m2) ** 2)) / (2 * big_m)


def boost(beta: torch.Tensor) -> torch.Tensor:
    r"""Λ^μ{}_ν for a frame moving with velocity -\boldsymbol β (so a particle at rest gets velocity +β).
    beta: (3,) → (4, 4)"""
    b2 = torch.dot(beta, beta)
    gamma = 1.0 / torch.sqrt(1.0 - b2)
    lam = torch.empty(4, 4, dtype=beta.dtype, device=beta.device)
    lam[0, 0] = gamma
    lam[0, 1:] = gamma * beta
    lam[1:, 0] = gamma * beta
    lam[1:, 1:] = torch.eye(3, dtype=beta.dtype, device=beta.device) + (gamma - 1) * torch.outer(beta, beta) / b2
    return lam


def transform(lam: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r"""p'^μ = Λ^μ{}_ν p^ν   (ν summed).   lam: (4, 4), p: (..., 4) → (..., 4)"""
    return torch.einsum("mn,...n->...m", lam, p)
