r"""Physics → tensor translation.

At fixed temperature T and pressure P, a reacting ideal-gas mixture settles where its Gibbs energy is lowest:

    G(n) = Σ_j n_j μ_j,      μ_j = μ°_j(T) + RT ln( n_j P / (n_tot P°) ),      n_tot = Σ_j n_j

subject to element conservation (atoms are neither made nor destroyed):

    A_{ij} n_j = b_i          (j summed: A_{ij} = atoms of element i in species j, b_i = atoms fed in)

Indices: b = batch (one temperature or one equivalence ratio per row), j/k = species, i/l = element.
At the minimum (Lagrange multipliers π_i, the "element potentials"):

    μ_{bj} / RT = π_{bi} A_{ij}       (i summed)

so every species is built from the same per-element chemical potentials. Dimensionless g = μ°/RT throughout.
"""

import math

import torch

from .tensors import T0, R


def standard_potential(T: torch.Tensor, th: dict) -> torch.Tensor:
    r"""g_{bj} = μ°_j(T_b) / RT_b = [H_j(T) - T S_j(T)] / RT with a constant c_p:   T (B,) → (B, S).

    H_j(T) = ΔfH°_j + c_{p,j} (T - T0),     S_j(T) = S°_j + c_{p,j} ln(T / T0)
    """
    T = T[:, None]  # (B, 1) broadcast against species (S,)
    H = 1e3 * th["dH"] + th["cp"] * (T - T0)  # J/mol
    S = th["S"] + th["cp"] * torch.log(T / T0)  # J/(mol K)
    return (H - T * S) / (R * T)


def chemical_potential(n: torch.Tensor, g: torch.Tensor, P: float) -> torch.Tensor:
    r"""μ_{bj}/RT = g_{bj} + ln(n_{bj} / n_tot,b) + ln(P/P°):   (B, S) → (B, S).   P in bar."""
    n_tot = n.sum(-1, keepdim=True)
    return g + torch.log(n / n_tot) + math.log(P)


def chemical_potential_log(ln_n: torch.Tensor, ln_ntot: torch.Tensor, g: torch.Tensor, P: float) -> torch.Tensor:
    """The same μ/RT from ln n_j and ln n_tot: the solver works with logarithms, so trace species of
    mole fraction 1e-40 stay representable."""
    return g + ln_n - ln_ntot[:, None] + math.log(P)


def gibbs(n: torch.Tensor, g: torch.Tensor, P: float) -> torch.Tensor:
    r"""G_b / RT = n_{bj} μ_{bj} / RT   (j summed):   (B, S) → (B,)."""
    return torch.einsum("bj,bj->b", n, chemical_potential(n, g, P))


def element_totals(A: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    r"""A_{ij} n_{bj}   (j summed): atoms of each element present,   (E, S), (B, S) → (B, E)."""
    return torch.einsum("ij,bj->bi", A, n)


def newton_system(A, n, n_tot, mu, b):
    r"""Linearised optimality conditions (the Gordon–McBride form of the RAND method), for unknowns
    π_{bi} and Δln n_tot:

        Σ_l [A_{ij} n_j A_{lj}] π_l + [A_{ij} n_j] Δln n_tot = b_i - A_{ij} n_j + A_{ij} n_j μ_j
        Σ_l [A_{lj} n_j]        π_l + [Σ_j n_j - n_tot] Δln n_tot = n_tot - Σ_j n_j + Σ_j n_j μ_j

    (j summed in every bracket). Returns J (B, E+1, E+1) and rhs (B, E+1).
    """
    M = torch.einsum("ij,bj,lj->bil", A, n, A)  # (B, E, E): j summed
    a = element_totals(A, n)  # (B, E)
    corner = n.sum(-1) - n_tot  # (B,)
    J = torch.cat(
        [torch.cat([M, a[:, :, None]], dim=2), torch.cat([a[:, None, :], corner[:, None, None]], dim=2)], dim=1
    )
    rhs_top = b - a + torch.einsum("ij,bj,bj->bi", A, n, mu)
    rhs_bot = n_tot - n.sum(-1) + torch.einsum("bj,bj->b", n, mu)
    return J, torch.cat([rhs_top, rhs_bot[:, None]], dim=1)


def species_update(A, mu, pi, dln_ntot):
    r"""Δln n_{bj} = -μ_{bj} + π_{bi} A_{ij} + Δln n_tot,b   (i summed):   → (B, S)."""
    return -mu + torch.einsum("bi,ij->bj", pi, A) + dln_ntot[:, None]


def enthalpy(n: torch.Tensor, T: torch.Tensor, th: dict) -> torch.Tensor:
    r"""H_b = n_{bj} [ΔfH°_j + c_{p,j}(T_b - T0)]   (j summed), in kJ:   (B, S), (B,) → (B,)."""
    h = th["dH"] + 1e-3 * th["cp"] * (T[:, None] - T0)  # (B, S) kJ/mol
    return torch.einsum("bj,bj->b", n, h)


def mole_fractions(n: torch.Tensor) -> torch.Tensor:
    """x_{bj} = n_{bj} / n_tot,b."""
    return n / n.sum(-1, keepdim=True)
