"""The solver: one batched eigh for all molecules, then fill the MOs and form bond orders and charges."""

import torch

from .equation import (
    bond_order_matrix,
    charges,
    hamiltonian,
    homo_lumo_gap,
    localized_energy,
    occupations,
    pi_energy,
)


def solve(A: torch.Tensor, pad: torch.Tensor, n_electrons: list[int], alpha: float, beta: float) -> dict:
    """Diagonalise H_bij = α δ_ij + β A_bij (+ padding) for the whole batch at once.

    Returns a dict of E (B, N) ascending, C (B, N, N) with MO k in column k, occupations n (B, N),
    E_pi, E_loc, delocalisation energy, HOMO–LUMO gap (B,), bond orders P (B, N, N) and charges q (B, N).
    """
    H = hamiltonian(A, alpha, beta) + pad  # (B, N, N)
    E, C = torch.linalg.eigh(H)  # (B, N), (B, N, N)
    n = occupations(E, n_electrons)
    E_pi = pi_energy(E, n)
    E_loc = localized_energy(n_electrons, alpha, beta).to(E.device)
    P = bond_order_matrix(C, n)
    return {
        "E": E,
        "C": C,
        "n": n,
        "E_pi": E_pi,
        "E_loc": E_loc,
        "deloc": E_pi - E_loc,
        "gap": homo_lumo_gap(E, n),
        "P": P,
        "q": charges(P),
    }
