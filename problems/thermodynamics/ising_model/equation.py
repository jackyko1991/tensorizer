r"""Physics → tensor translation.

The 2D Ising model: spins s_{ij} = ±1 on an L × L square lattice (periodic edges) with energy

    E = -J \sum_{\langle ij, kl \rangle} s_{ij} s_{kl}        (sum over nearest-neighbour pairs)

Write h_{ij} for the sum of the four neighbours of site (i, j). That is a convolution with a cross-shaped kernel,

    h_{ij} = \sum_{kl} K_{kl} \, s_{i+k, j+l},     K = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

so conv2d computes it for every site at once (circular padding makes the edges periodic). Then

    E = -\tfrac{J}{2} \, s_{ij} h_{ij}         (repeated i, j summed; ½ because every bond is seen twice)
    m = \tfrac{1}{N} \sum_{ij} s_{ij}          (magnetisation per spin, N = L²)

Metropolis: flipping s_{ij} costs \Delta E_{ij} = 2 J s_{ij} h_{ij}; accept the flip with probability
min(1, e^{-\Delta E / T}) (units with k_B = 1). Sites of one checkerboard colour have no neighbours of the same
colour, so all of them can be updated together without disturbing each other's h_{ij}.

Onsager's exact result: below T_c = 2J / \ln(1 + \sqrt2) ≈ 2.269 J the spontaneous magnetisation is
|m| = (1 - \sinh^{-4}(2J/T))^{1/8}, and m = 0 above T_c.
"""

import math

import torch
import torch.nn.functional as F

T_C = 2 / math.log(1 + math.sqrt(2))  # critical temperature for J = 1, k_B = 1
KERNEL = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # K_{kl}: the four neighbours


def neighbor_sum(s: torch.Tensor) -> torch.Tensor:
    """h_{ij} = Σ_kl K_kl s_{i+k, j+l} with periodic edges.   s: (B, 1, L, L) → (B, 1, L, L)"""
    padded = F.pad(s, (1, 1, 1, 1), mode="circular")  # (B, 1, L+2, L+2)  wrap the edges around
    kernel = KERNEL.to(s.device, s.dtype)[None, None]  # (1, 1, 3, 3)  out channels, in channels, 3, 3
    return F.conv2d(padded, kernel)


def energy_per_spin(s: torch.Tensor, J: float = 1.0) -> torch.Tensor:
    r"""E / N = −(J/2) s_{ij} h_{ij} / N.   (B, 1, L, L) → (B,)"""
    h = neighbor_sum(s)
    n_sites = s.shape[-1] * s.shape[-2]
    return -0.5 * J * torch.einsum("bcij,bcij->b", s, h) / n_sites


def magnetization(s: torch.Tensor) -> torch.Tensor:
    """m = average spin.   (B, 1, L, L) → (B,)"""
    return s.mean(dim=(1, 2, 3))


def flip_cost(s: torch.Tensor, h: torch.Tensor, J: float = 1.0) -> torch.Tensor:
    """ΔE_{ij} = 2 J s_{ij} h_{ij}: energy change if spin (i, j) alone were flipped.   (B, 1, L, L)"""
    return 2 * J * s * h


def onsager_magnetization(T: torch.Tensor, J: float = 1.0) -> torch.Tensor:
    """Exact spontaneous magnetisation of the infinite lattice: (1 − sinh(2J/T)⁻⁴)^(1/8) below T_c, else 0."""
    inside = 1 - torch.sinh(2 * J / T) ** -4
    return torch.where(T < T_C, inside.clamp(min=0) ** 0.125, torch.zeros_like(T))
