r"""Chemistry → tensor translation.

A reaction with signed stoichiometric numbers ν_s (negative for reactants) changes every amount by the same
reaction extent ξ (in mol):

    n_{bs}(ξ) = n⁰_{bs} + ν_s ξ_b                       (experiment b, species s; no index summed)

The reaction stops when the first reactant runs out — the limiting reagent:

    ξ^max_b = min_{s: ν_s < 0}  n⁰_{bs} / |ν_s|

Theoretical yield of the product p: n_{bp} = ν_p ξ^max_b, i.e. a mass M_p ν_p ξ^max_b;
percent yield = weighed / theoretical.
"""

import torch


def moles(mass: torch.Tensor, molar: torch.Tensor) -> torch.Tensor:
    """n_{bs} = m_{bs} / M_s.   (B, S), (S,) → (B, S)."""
    return mass / molar


def amounts(n0: torch.Tensor, nu: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """n_{b…s}(ξ) = n⁰_{bs} + ν_s ξ_{b…}.   n0: (B, S), ν: (S,), ξ: (B, …) → (B, …, S).  No index summed."""
    shape = (n0.shape[0],) + (1,) * (xi.dim() - 1) + (n0.shape[-1],)
    return n0.reshape(shape) + nu * xi[..., None]


def max_extent(n0: torch.Tensor, nu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ξ^max = min over reactants of n⁰_s / |ν_s|, and which reactant reaches zero first.   (…, S) → (…), (…).

    Products (ν_s > 0) never run out, so they get +∞ and can never be the minimum.
    """
    ratio = torch.where(nu < 0, n0 / nu.abs(), torch.inf)  # (…, S) how many "reaction units" each reactant allows
    xi_max, limiting = ratio.min(dim=-1)
    return xi_max, limiting


def theoretical_mass(xi_max: torch.Tensor, nu: torch.Tensor, molar: torch.Tensor, product: int = -1) -> torch.Tensor:
    """Most product the reaction can make: m_p = M_p ν_p ξ^max (g).   (…) → (…)."""
    return molar[product] * nu[product] * xi_max


def percent_yield(weighed: torch.Tensor, theoretical: torch.Tensor) -> torch.Tensor:
    """Actual over theoretical, in %.   (B,) → (B,)."""
    return 100.0 * weighed / theoretical
