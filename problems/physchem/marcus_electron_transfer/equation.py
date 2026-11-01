r"""Physics → tensor translation: Marcus parabolas, the Marcus rate and the Landau–Zener hopping probability.

Indices: l reorganisation energy, t temperature, g driving force, d simulated driving force, n trajectory.
No index is summed here: every formula is evaluated on a broadcast grid, e.g. k_ltg for all λ_l, T_t, ΔG°_g at once.

    G_R(x) = λ x²,   G_P(x) = λ (x - 1)² + ΔG°                     free-energy surfaces of the solvent coordinate
    x* = (λ + ΔG°) / 2λ,   ΔG‡ = G_R(x*) = (ΔG° + λ)² / 4λ           where the parabolas cross
    k_ltg = (2π/ħ) |H_AB|² (4π λ_l k_B T_t)^{-1/2} exp(-(ΔG°_g + λ_l)² / 4 λ_l k_B T_t)
    P_LZ = 1 - exp(-2π |H_AB|² / (ħ |ẋ| |G_P' - G_R'|)),  |G_P' - G_R'| = 2λ
"""

import math

import torch

from .tensors import HBAR


def reactant_surface(x: torch.Tensor, lam) -> torch.Tensor:
    """G_R(x) = λ x² (eV)."""
    return lam * x**2


def product_surface(x: torch.Tensor, lam, dG) -> torch.Tensor:
    """G_P(x) = λ (x - 1)² + ΔG° (eV)."""
    return lam * (x - 1) ** 2 + dG


def crossing_point(lam, dG):
    """x* where G_R = G_P: λx² = λ(x-1)² + ΔG°  ⇒  x* = (λ + ΔG°) / 2λ."""
    return (lam + dG) / (2 * lam)


def activation_energy(lam, dG):
    """ΔG‡ = G_R(x*) = (ΔG° + λ)² / 4λ: zero when -ΔG° = λ, rising again beyond (the inverted region)."""
    return (dG + lam) ** 2 / (4 * lam)


def marcus_rate(dG: torch.Tensor, lam: torch.Tensor, kT: torch.Tensor, H_ab: float) -> torch.Tensor:
    """Non-adiabatic Marcus rate (1/ps). Broadcasts: e.g. dG (1, 1, G), lam (L, 1, 1), kT (1, T, 1) → (L, T, G)."""
    prefactor = 2 * math.pi / HBAR * H_ab**2 / torch.sqrt(4 * math.pi * lam * kT)
    return prefactor * torch.exp(-activation_energy(lam, dG) / kT)


def landau_zener(speed: torch.Tensor, lam, H_ab: float) -> torch.Tensor:
    """Probability of switching from one diabatic surface to the other in one passage of the crossing.

    speed |ẋ| in 1/ps; the slopes of the parabolas differ by 2λ everywhere (same curvature).
    """
    return 1 - torch.exp(-2 * math.pi * H_ab**2 / (HBAR * speed.clamp_min(1e-12) * 2 * lam))


def solvent_force(x: torch.Tensor, state: torch.Tensor, omega: float) -> torch.Tensor:
    """Acceleration on the current surface: ẍ = -ω² (x - x_min), x_min = 0 on the reactant, 1 on the product."""
    return -(omega**2) * (x - state)
