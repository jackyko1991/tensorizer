r"""Physics → tensor translation: Hückel molecular-orbital theory for π electrons.

Indices: b molecule, i and j carbon atoms, k molecular orbital (MO).

Each carbon contributes one 2p orbital perpendicular to the molecular plane. In that basis the π Hamiltonian is
    H_bij = α δ_ij + β A_bij,
α the energy of an electron in an isolated 2p orbital, β < 0 the resonance (hopping) integral between bonded
neighbours and A the adjacency matrix of the carbon skeleton. Diagonalising, H C = C diag(E), gives the MO
energies E_bk and the coefficients C_bik of atom i in MO k.

With n_bk electrons in MO k (0, 1 or 2):
    E_π       = n_bk E_bk                         (k summed)
    P_bij     = n_bk C_bik C_bjk                  (k summed): Coulson bond order on a bond, π population on i = j
    q_bi      = 1 - P_bii                         π charge of atom i in a neutral molecule
For a ring of N atoms the eigenvalues are known in closed form, E_k = α + 2β cos(2πk / N), k = 0 … N - 1.
"""

import math

import torch


def hamiltonian(A: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """H_bij = α δ_ij + β A_bij.   A (B, N, N) → (B, N, N)"""
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return alpha * eye + beta * A


def occupations(E: torch.Tensor, n_electrons: list[int], tol: float = 1e-6) -> torch.Tensor:
    """Aufbau filling n_bk: two electrons per MO from the lowest energy up. A partly filled set of degenerate MOs
    shares its electrons equally (Hund-like averaging), so the result does not depend on how eigh rotates them.

    E (B, N) ascending, n_electrons[b] → (B, N)"""
    n = torch.zeros_like(E)
    for b, left in enumerate(n_electrons):
        k = 0
        while left > 0:
            shell = (E[b] - E[b, k]).abs() < tol  # this level and every MO degenerate with it
            g = int(shell.sum())
            take = min(left, 2 * g)
            n[b, shell] = take / g
            left -= take
            k += g
    return n


def pi_energy(E: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """E_π = n_bk E_bk, k summed.   (B,)"""
    return torch.einsum("bk,bk->b", n, E)


def localized_energy(n_electrons: list[int], alpha: float, beta: float) -> torch.Tensor:
    """Energy of isolated ethylene-like double bonds (2 electrons at α + β each) plus a lone electron at α
    if the count is odd: the reference for the delocalisation energy.   (B,)"""
    return torch.tensor([(ne // 2) * 2 * (alpha + beta) + (ne % 2) * alpha for ne in n_electrons], dtype=torch.float64)


def bond_order_matrix(C: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """P_bij = n_bk C_bik C_bjk, k summed over the occupied MOs.   (B, N, N)"""
    return torch.einsum("bk,bik,bjk->bij", n, C, C)


def charges(P: torch.Tensor) -> torch.Tensor:
    """q_bi = 1 - P_bii: zero for every atom of an alternant hydrocarbon (Coulson–Rushbrooke).   (B, N)"""
    return 1 - torch.diagonal(P, dim1=-2, dim2=-1)


def homo_lumo_gap(E: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """LUMO minus HOMO energy: lowest MO with room for an electron minus highest MO holding one.   (B,)"""
    big = torch.finfo(E.dtype).max
    homo = torch.where(n > 0, E, -big).max(-1).values
    lumo = torch.where(n < 2, E, big).min(-1).values
    return lumo - homo


def cyclic_levels(N: int, alpha: float, beta: float) -> torch.Tensor:
    """Closed-form eigenvalues of a ring of N atoms, E_k = α + 2β cos(2πk/N), sorted ascending.   (N,)"""
    k = torch.arange(N, dtype=torch.float64)
    return (alpha + 2 * beta * torch.cos(2 * math.pi * k / N)).sort().values


def chain_levels(N: int, alpha: float, beta: float) -> torch.Tensor:
    """Closed-form eigenvalues of an open chain of N atoms, E_k = α + 2β cos(πk/(N+1)), k = 1 … N.   (N,)"""
    k = torch.arange(1, N + 1, dtype=torch.float64)
    return (alpha + 2 * beta * torch.cos(math.pi * k / (N + 1))).sort().values


def is_closed_shell(levels: torch.Tensor, n_electrons: int, tol: float = 1e-6) -> bool:
    """True when the electrons exactly fill a set of degenerate levels: a gap separates HOMO and LUMO.
    For a ring this happens for 4n + 2 electrons (Hückel's rule); 4n leaves a degenerate pair half filled."""
    n = occupations(levels[None], [n_electrons], tol)[0]
    return bool(homo_lumo_gap(levels[None], n[None])[0] > tol)
