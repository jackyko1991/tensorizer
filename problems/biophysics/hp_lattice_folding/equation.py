r"""Physics → tensor translation for the HP lattice protein.

A protein is a chain of N residues on the square lattice, x_{rik} = coordinate k of residue i in replica r.
Residues are either hydrophobic (h_i = 1) or polar (h_i = 0). Two residues that touch on the lattice but are not
neighbours along the chain form a contact, and every H–H contact lowers the energy by ε = 1:

    d_{rij} = (x_{rik} - x_{rjk}) (x_{rik} - x_{rjk})        summed over k:   squared distance
    C_{rij} = [d_{rij} = 1] [|i - j| > 1]                     contact map
    E_r     = -½ h_i C_{rij} h_j                              summed over i and j (½: each pair counted twice)

A chain is allowed only if consecutive residues are one lattice step apart and no two residues share a site
(d_{rij} > 0 for i ≠ j): a self-avoiding walk.
"""

import torch


def squared_distances(x: torch.Tensor) -> torch.Tensor:
    """d_{rij} = |x_ri|² + |x_rj|² - 2 x_{rik} x_{rjk}, repeated k summed.   x: (R, N, 2) → (R, N, N).

    This is (x_ri - x_rj)·(x_ri - x_rj) expanded, so the pair term is one batched matrix product.
    """
    x = x - x[:, :1]  # put residue 0 at the origin: distances don't change and the numbers stay small
    norm = torch.einsum("rik,rik->ri", x, x)  # (R, N)
    gram = torch.einsum("rik,rjk->rij", x, x)  # (R, N, N)
    return norm[:, :, None] + norm[:, None, :] - 2 * gram


def contact_map(d: torch.Tensor) -> torch.Tensor:
    """C_{rij} = [d_{rij} = 1] [|i - j| > 1]: lattice neighbours that are not chain neighbours.   (R, N, N)."""
    n = d.shape[1]
    idx = torch.arange(n, device=d.device)
    non_bonded = (idx[:, None] - idx[None, :]).abs() > 1  # (N, N)
    return ((d == 1) & non_bonded).float()


def energy(C: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """E_r = -½ h_i C_{rij} h_j, repeated i and j summed.   C: (R, N, N), h: (N,) → (R,)."""
    return -0.5 * torch.einsum("i,rij,j->r", h, C, h)


def chain_energy(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Energy straight from the coordinates: x → d → C → E.   (R,)."""
    return energy(contact_map(squared_distances(x)), h)


def self_avoiding(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """True for replicas whose bonds all have length 1 and whose residues all sit on different sites.   (R,) bool.

    x: (R, N, 2) coordinates, d: (R, N, N) their squared distances.
    """
    n = x.shape[1]
    bonds = (x[:, 1:] - x[:, :-1]).abs().sum(-1)  # (R, N-1) lattice step lengths
    overlap = (d == 0) & ~torch.eye(n, dtype=torch.bool, device=x.device)
    return (bonds == 1).all(-1) & ~overlap.any(-1).any(-1)


def metropolis(dE: torch.Tensor, T: float, u: torch.Tensor) -> torch.Tensor:
    """Accept a move with probability min(1, exp(-ΔE / T)).   dE, u: (R,) with u uniform in [0, 1)."""
    return u < torch.exp(-dE / T)
