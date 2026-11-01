"""Build the concrete tensors: Pauli matrices, gate tensors, basis states and the Ising Hamiltonian."""

import math

import torch

C = torch.complex128


def pauli(device: torch.device) -> torch.Tensor:
    """σ_{kab}: the three Pauli matrices stacked along k ∈ {x, y, z}. Shape (3, 2, 2) complex."""
    sx = [[0, 1], [1, 0]]
    sy = [[0, -1j], [1j, 0]]
    sz = [[1, 0], [0, -1]]
    return torch.tensor([sx, sy, sz], dtype=C, device=device)


def hadamard(device: torch.device) -> torch.Tensor:
    """Single-qubit Hadamard H = (X + Z)/√2. Shape (2, 2)."""
    return torch.tensor([[1, 1], [1, -1]], dtype=C, device=device) / math.sqrt(2)


def on_qubit_1(A: torch.Tensor) -> torch.Tensor:
    """(A ⊗ I)_{abcd} = A_{ac} δ_{bd}: act with A on qubit 1, leave qubit 2 alone. Shape (2, 2, 2, 2)."""
    return torch.einsum("ac,bd->abcd", A, torch.eye(2, dtype=A.dtype, device=A.device))


def on_qubit_2(B: torch.Tensor) -> torch.Tensor:
    """(I ⊗ B)_{abcd} = δ_{ac} B_{bd}. Shape (2, 2, 2, 2)."""
    return torch.einsum("ac,bd->abcd", torch.eye(2, dtype=B.dtype, device=B.device), B)


def cnot(device: torch.device) -> torch.Tensor:
    """CNOT with qubit 1 as control: G_{abcd} = 1 if a = c and b = c XOR d. Shape (2, 2, 2, 2)."""
    G = torch.zeros(2, 2, 2, 2, dtype=C, device=device)
    for c in range(2):
        for d in range(2):
            G[c, c ^ d, c, d] = 1.0  # |c, d⟩ → |c, c ⊕ d⟩
    return G


def basis_state(a: int, b: int, device: torch.device) -> torch.Tensor:
    """|a b⟩ as a tensor: ψ_{a'b'} = δ_{aa'} δ_{bb'}. Shape (2, 2)."""
    psi = torch.zeros(2, 2, dtype=C, device=device)
    psi[a, b] = 1.0
    return psi


def ising_hamiltonian(J: float, h: float, sigma: torch.Tensor) -> torch.Tensor:
    """H_{abcd} = J Z_{ac} Z_{bd} + h (X_{ac} δ_{bd} + δ_{ac} X_{bd}). Shape (2, 2, 2, 2).

    Reshaped to (4, 4) this equals torch.kron(Z, Z) * J + h * (torch.kron(X, I) + torch.kron(I, X)).
    """
    X, Z = sigma[0], sigma[2]
    zz = torch.einsum("ac,bd->abcd", Z, Z)  # Z ⊗ Z
    return J * zz + h * (on_qubit_1(X) + on_qubit_2(X))
