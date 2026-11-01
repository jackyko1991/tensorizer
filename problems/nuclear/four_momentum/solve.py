"""Boost every decay configuration to the lab and reconstruct the parent from its products."""

import torch

from .equation import boost, invariant_mass, transform


def to_lab(parent: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the boost with the parent's velocity β = P/E to both daughters: (B, 4) each."""
    lam = boost(parent[1:] / parent[0])  # (4, 4)
    return transform(lam, p1), transform(lam, p2)


def reconstruct(p1: torch.Tensor, p2: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Invariant mass of the pair, √((p1 + p2)·(p1 + p2)), for every configuration: shape (B,)."""
    return invariant_mass(p1 + p2, eta)
