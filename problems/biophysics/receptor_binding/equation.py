r"""Physics → tensor translation.

A ligand L (an antigen, or a peptide–MHC) binds a receptor R on an immune cell and forms a complex C:

    R + L  ⇌  C        forward rate k_on R L,   backward rate k_off C

For a batch of receptor variants b and a set of ligand concentrations c the net binding flux is

    J_{bc} = k^{on}_b R_{bc} L_{bc} - k^{off}_b C_{bc}

and the two species change by the same amount in opposite directions:

    \dot C_{bc} = J_{bc},    \dot R_{bc} = -J_{bc}      ⇒   R_{bc} + C_{bc} = R_T  (receptors are conserved)

No index is summed here: every (b, c) pair is an independent experiment. The einsum calls below only
make the broadcasting explicit — a rate constant k_b is shared by every concentration c.
"""

import math

import torch


def free_ligand(L_total: torch.Tensor, C: torch.Tensor, depletion: bool) -> torch.Tensor:
    """L_{bc} = L_c - δ C_{bc}.   L_total: (C,), C: (B, C) → (B, C).

    δ = 0: ligand is in large excess (or keeps flowing in, as in SPR), so L_{bc} = L_c.
    δ = 1: ligand depletion — every complex formed removes one ligand from solution.
    """
    if depletion:
        return L_total - C  # (C,) broadcast over b, minus (B, C)
    return L_total.expand_as(C)


def flux(R: torch.Tensor, C: torch.Tensor, L: torch.Tensor, k_on: torch.Tensor, k_off: torch.Tensor) -> torch.Tensor:
    r"""J_{bc} = k^{on}_b R_{bc} L_{bc} - k^{off}_b C_{bc}.   R, C, L: (B, C);  k_on, k_off: (B,)."""
    binding = torch.einsum("b,bc,bc->bc", k_on, R, L)  # association: needs a free receptor AND a free ligand
    unbinding = torch.einsum("b,bc->bc", k_off, C)  # dissociation: each complex falls apart on its own
    return binding - unbinding


def rhs(y: torch.Tensor, L_total: torch.Tensor, k_on: torch.Tensor, k_off: torch.Tensor, depletion: bool):
    """dy/dt for the state y of shape (2, B, C): y[0] = free receptors R, y[1] = complexes C."""
    R, C = y[0], y[1]
    L = free_ligand(L_total, C, depletion)
    J = flux(R, C, L, k_on, k_off)
    return torch.stack([-J, J])


def dissociation_constant(k_on: torch.Tensor, k_off: torch.Tensor) -> torch.Tensor:
    """K_D = k_off / k_on (a concentration): the ligand level that fills half the receptors.   (B,)"""
    return k_off / k_on


def half_life(k_off: torch.Tensor) -> torch.Tensor:
    """t_½ = ln 2 / k_off: how long a complex lasts on average before half of them have fallen apart.   (B,)"""
    return math.log(2.0) / k_off


def observed_rate(L: torch.Tensor, k_on: torch.Tensor, k_off: torch.Tensor) -> torch.Tensor:
    """k_obs,bc = k^on_b L_c + k^off_b: the rate at which binding approaches equilibrium.   (B, C)"""
    return torch.einsum("b,c->bc", k_on, L) + k_off[:, None]


def occupancy(L: torch.Tensor, K_D: torch.Tensor) -> torch.Tensor:
    """Equilibrium fraction of receptors bound, θ_bc = L_c / (L_c + K_D,b)  (ligand in excess).   (B, C)"""
    return L[None, :] / (L[None, :] + K_D[:, None])


def occupancy_with_depletion(L_total: torch.Tensor, K_D: torch.Tensor, R_T: float) -> torch.Tensor:
    """Equilibrium θ_bc when ligand is depleted: k_on (R_T - C)(L_T - C) = k_off C, a quadratic in C.

    C² - s C + R_T L_T = 0  with  s = R_T + L_T + K_D   →   the smaller root (the physical one, C ≤ R_T, L_T):

    C = (s - √(s² - 4 R_T L_T)) / 2 = 2 R_T L_T / (s + √(s² - 4 R_T L_T))

    The second form is the same number but avoids subtracting two nearly equal floats.
    """
    s = R_T + L_total[None, :] + K_D[:, None]  # (B, C)
    C = 2.0 * R_T * L_total[None, :] / (s + torch.sqrt(s**2 - 4.0 * R_T * L_total[None, :]))
    return C / R_T
