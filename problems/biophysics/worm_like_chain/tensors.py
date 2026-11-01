"""Build the concrete tensors: the applied forces, the bending stiffness that gives the wanted persistence length,
and the starting chains. Units: lengths in nm, forces in pN, energies in k_B T (k_B T = 4.11 pN·nm).
"""

import math

import torch

from .equation import KT, persistence_from_stiffness

SEGMENT = 2.5  # segment length a in nm: much shorter than P, so the discrete chain behaves like a continuous WLC


def forces(n: int, device: torch.device) -> torch.Tensor:
    """F_b: F_0 = 0 (a free chain, used for the tangent correlation), then n forces spread evenly on a log scale
    from 0.02 pN to 10 pN, like an optical- or magnetic-tweezers run.   Shape (B,) = (n + 1,)."""
    pulled = torch.logspace(math.log10(0.02), 1.0, n, device=device)
    return torch.cat([torch.zeros(1, device=device), pulled])


def reduced_forces(F: torch.Tensor, a: float) -> torch.Tensor:
    """f_b = F_b a / k_B T: the work done by the force on one segment that points along it.   (B,)"""
    return F * a / KT


def stiffness_for(P: float, a: float) -> float:
    """Bending stiffness K = κ / k_B T for which the discrete chain has persistence length P exactly.

    P(K) = -a / ln(coth K - 1/K) increases with K, so a bisection finds it; K ≈ P/a + 1/2 for P ≫ a.
    """
    lo, hi = 1e-3, 10 * P / a + 10
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if persistence_from_stiffness(mid, a) < P:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def relaxed_chains(n_forces: int, n_chains: int, n_segments: int, K: float, seed: int, device: torch.device):
    """t_{bmik}: chains drawn exactly from the force-free worm-like chain, so the run starts close to equilibrium.

    Without force each tangent depends only on the one before: the angle θ to it follows p(cos θ) ∝ exp(K cos θ),
    sampled by inverting its cumulative distribution, in a random direction around it.   (B, M, N, 3)
    """
    gen = torch.Generator().manual_seed(seed)
    shape = (n_forces, n_chains)
    t = torch.nn.functional.normalize(torch.randn(*shape, 3, generator=gen), dim=-1)  # first tangent: random
    chain = [t]
    for _ in range(n_segments - 1):
        u = torch.rand(shape, generator=gen)
        cos = 1 + torch.log(u + (1 - u) * math.exp(-2 * K)) / K  # (B, M), cos θ between t_i and t_{i+1}
        g = torch.randn(*shape, 3, generator=gen)
        side = torch.nn.functional.normalize(g - (g * t).sum(-1, keepdim=True) * t, dim=-1)  # random ⟂ t_i
        t = cos[..., None] * t + torch.sqrt((1 - cos**2).clamp(min=0))[..., None] * side
        t = torch.nn.functional.normalize(t, dim=-1)  # remove round-off so |t| stays exactly 1
        chain.append(t)
    return torch.stack(chain, dim=2).to(device)
