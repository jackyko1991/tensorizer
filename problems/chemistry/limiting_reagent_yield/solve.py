"""Stoichiometry for a whole batch at once: limiting reagent, yields, the extent sweep and the yield map."""

import torch

from . import equation


def analyse(m0: torch.Tensor, weighed: torch.Tensor, nu: torch.Tensor, molar: torch.Tensor) -> dict:
    """Everything for the batch of experiments: m0 (B, S) starting masses, weighed (B,) product masses."""
    n0 = equation.moles(m0, molar)  # (B, S)
    xi_max, limiting = equation.max_extent(n0, nu)  # (B,), (B,)
    n_end = equation.amounts(n0, nu, xi_max)  # (B, S): the limiting reagent is exactly 0
    theory = equation.theoretical_mass(xi_max, nu, molar)  # (B,)
    return {
        "n0": n0,
        "xi_max": xi_max,
        "limiting": limiting,
        "n_end": n_end,
        "leftover_mass": n_end * molar,  # (B, S) g of each species when the reaction stops
        "theory": theory,
        "percent": equation.percent_yield(weighed, theory),
    }


def extent_sweep(n0: torch.Tensor, nu: torch.Tensor, xi_max: torch.Tensor, steps: int = 5):
    """Amounts at ξ = 0, ¼, ½, ¾, 1 × ξ^max for every experiment.   → ξ (B, K), n (B, K, S)."""
    frac = torch.linspace(0.0, 1.0, steps, dtype=n0.dtype, device=n0.device)  # (K,)
    xi = xi_max[:, None] * frac  # (B, K)
    return xi, equation.amounts(n0, nu, xi)


def yield_map(mass_1: torch.Tensor, mass_2: torch.Tensor, nu: torch.Tensor, molar: torch.Tensor) -> torch.Tensor:
    """Theoretical product mass over a grid of the two reactant masses.   (G1,), (G2,) → (G2, G1) in g.

    The grid is just a big batch: n⁰ has shape (G2, G1, S) and the same min-over-reactants does the rest.
    """
    m2, m1 = torch.meshgrid(mass_2, mass_1, indexing="ij")  # (G2, G1)
    m0 = torch.stack([m1, m2, torch.zeros_like(m1)], dim=-1)  # (G2, G1, S)
    xi_max, _ = equation.max_extent(equation.moles(m0, molar), nu)
    return equation.theoretical_mass(xi_max, nu, molar)
