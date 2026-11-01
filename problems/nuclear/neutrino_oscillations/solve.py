"""Evaluate the oscillation probabilities on the whole baseline grid at once."""

import torch

from .equation import phases, probability


def oscillate(u: torch.Tensor, m2: torch.Tensor, dist: torch.Tensor, energy: float, alpha: int) -> torch.Tensor:
    """P(α → β) for a beam born as flavour α: shape (K, 3) with columns β = e, μ, τ."""
    p = probability(u.to(dist.device), phases(m2, dist / energy))  # (K, β, α)
    return p[:, :, alpha]


def detected_flavour(p_at: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Sample a flavour per neutrino: the first β whose cumulative probability exceeds u_n.

    p_at: (n, 3) probabilities at each neutrino's position, u: (n,) → (n,) integer flavour index.
    """
    cumulative = torch.cumsum(p_at, dim=1)  # (n, 3)
    return (cumulative < u[:, None]).sum(dim=1).clamp(max=2)
