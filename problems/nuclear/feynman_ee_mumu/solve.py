"""Evaluate the diagram on an (energy × angle) grid and integrate it into the total cross section."""

import math

import torch

from problems._common.progress import steps

from .equation import amplitude_squared, dsigma_domega
from .tensors import M_E, momenta


def cross_sections(sqrt_s, theta, m_f: float, alpha: float, gamma, eta, chunk: int = 16):
    """dσ/dΩ (E, B) in nb/sr and σ (E,) in nb = 2π ∫ dσ/dΩ sin θ dθ (trapezoid rule)."""
    rows = []
    for i in steps(math.ceil(sqrt_s.shape[0] / chunk), "energy chunks"):
        e = sqrt_s[i * chunk : (i + 1) * chunk]
        p, pb, k, kb = momenta(e, theta, m_f)  # (e, B, 4) each
        msq = amplitude_squared(p, pb, k, kb, M_E, m_f, alpha, gamma, eta)  # (e, B)
        rows.append(dsigma_domega(msq, e, M_E, m_f))
    dsdo = torch.cat(rows)
    sigma = 2 * math.pi * torch.trapezoid(dsdo * torch.sin(theta), theta, dim=-1)
    return dsdo, sigma


def forward_backward(dsdo: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """A_FB = (σ_F − σ_B)/(σ_F + σ_B): zero for pure QED (the 1 + cos²θ shape is symmetric). (E, B) → (E,)"""
    w = dsdo * torch.sin(theta)
    mid = theta.shape[0] // 2  # θ grid has an odd number of points, so theta[mid] = 90° closes both halves
    f = torch.trapezoid(w[..., : mid + 1], theta[: mid + 1], dim=-1)
    b = torch.trapezoid(w[..., mid:], theta[mid:], dim=-1)
    return (f - b) / (f + b)
