"""Diagonalise the finite-difference box Hamiltonian of every molecule at once with torch.linalg.eigh."""

import torch


def solve(H: torch.Tensor, dx: torch.Tensor, n_levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowest levels E (B, n) and normalised wavefunctions ψ (B, M, n) of the batched Hamiltonian H (B, M, M).

    eigh treats the leading dimension as a batch: one call diagonalises every box.
    Column n of ψ_b is state n + 1; it is scaled so that Σ_i ψ_bin² dx_b = 1.
    """
    # float64: the largest grid eigenvalue is ~10⁵ × the lowest, too much spread for float32 round-off
    E, psi = torch.linalg.eigh(H.double())  # ascending energies
    E, psi = E[:, :n_levels].to(H.dtype), psi[:, :, :n_levels].to(H.dtype)
    psi = psi / torch.sqrt(dx)[:, None, None]  # unit vectors → Σ ψ² dx = 1
    sign = torch.sign(psi[:, :1, :])  # fix the arbitrary sign: every state starts upward at the left wall
    return E, psi * sign
