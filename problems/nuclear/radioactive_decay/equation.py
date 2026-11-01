r"""Physics → tensor translation.

Every unstable nucleus has the same chance per unit time λ to decay, so the expected number left obeys

    dN/dt = -λ N    ⇒    N(t) = N_0 e^{-λ t},    half-life  T_{1/2} = \ln 2 / λ.

In a short step Δt one nucleus decays with probability p = 1 - e^{-λ Δt}. With s_n = 1 (still there) or 0
(decayed) for nucleus n and a uniform random number u_n ∈ [0, 1):

    s_n(t + Δt) = s_n(t) · [u_n ≥ p],        N(t) = s_n(t)   summed over n.
"""

import math

import torch


def decay_constant(half_life: float) -> float:
    r"""λ = \ln 2 / T_{1/2}."""
    return math.log(2.0) / half_life


def exact_count(n0: int, lam: float, t: torch.Tensor) -> torch.Tensor:
    r"""N(t) = N_0 e^{-λ t}.   t: (T,) → (T,)"""
    return n0 * torch.exp(-lam * t)


def decay_probability(lam: float, dt: float) -> float:
    r"""p = 1 - e^{-λ Δt}: chance that one nucleus decays during one step."""
    return 1.0 - math.exp(-lam * dt)


def decay_step(alive: torch.Tensor, u: torch.Tensor, p: float) -> torch.Tensor:
    r"""s_n(t + Δt) = s_n(t) · [u_n ≥ p].   alive, u: same shape (any) → bool tensor of that shape"""
    return alive & (u >= p)


def count(alive: torch.Tensor) -> torch.Tensor:
    r"""N = s_n summed over the nuclei (the last two axes form the grid).   (..., S, S) → (...,)"""
    return alive.sum(dim=(-2, -1))
