"""Estimate π both ways for every experiment and measure how the error shrinks with N."""

import torch

from .equation import hits, needle_crosses, running_estimate


def darts_estimate(x: torch.Tensor) -> torch.Tensor:
    """Running π estimate from the darts, shape (B, N)."""
    return running_estimate(hits(x), 4.0)


def needle_estimate(y: torch.Tensor, phi: torch.Tensor, length: float) -> torch.Tensor:
    """π ≈ 2ℓN / (d · crossings), shape (B,); inf if no needle crossed."""
    crossings = needle_crosses(y, phi, length).sum(dim=-1)
    return 2 * length * y.shape[-1] / crossings


def spread(estimates: torch.Tensor) -> torch.Tensor:
    """Standard deviation across the B experiments at every N: (B, N) → (N,)."""
    return estimates.std(dim=0)
