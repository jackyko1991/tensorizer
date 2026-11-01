"""Diagonalise H, build U(t) = e^{−iHt} with matrix_exp, and evolve ⟨X⟩, ⟨P⟩ in both pictures."""

import torch

from .equation import expectation, heisenberg, schrodinger


def energies(H: torch.Tensor) -> torch.Tensor:
    """Eigenvalues of the Hermitian H, ascending: (N,)."""
    return torch.linalg.eigvalsh(H)


def propagators(H: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """U(t) = exp(−iHt) for every time at once (batched matrix exponential). (T,) → (T, N, N)"""
    return torch.linalg.matrix_exp(-1j * times[:, None, None].to(H.dtype) * H)


def heisenberg_picture(U: torch.Tensor, ops: dict, c: torch.Tensor) -> tuple[dict, dict]:
    """Operators move, the state stays: ⟨O⟩(t) = c† (U†OU) c.

    Returns ({name: ⟨O⟩ (T,)}, {name: O(t) (T, N, N)}).
    """
    moving = {name: heisenberg(U, op) for name, op in ops.items()}  # (T, N, N) each
    return {name: expectation(c, Ot) for name, Ot in moving.items()}, moving


def schrodinger_picture(U: torch.Tensor, ops: dict, c: torch.Tensor) -> dict:
    """The state moves, operators stay: ⟨O⟩(t) = c(t)† O c(t) with c(t) = U c. Returns {name: (T,)}."""
    ct = schrodinger(U, c)  # (T, N)
    return {name: expectation(ct, op) for name, op in ops.items()}


def first_order_energies(n: torch.Tensor, lam: float) -> torch.Tensor:
    """Perturbation theory for H₀ + λX⁴: E_n ≈ n + ½ + λ⟨n|X⁴|n⟩ = n + ½ + λ·¾(2n² + 2n + 1)."""
    return n + 0.5 + lam * 0.75 * (2 * n**2 + 2 * n + 1)
