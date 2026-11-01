"""Build the concrete matrices in the number basis |0⟩ … |N−1⟩ and the coherent start state."""

import math

import torch


def lowering(N: int, device: torch.device) -> torch.Tensor:
    """a_{mn} = √n δ_{m,n−1}: √1, √2, …, √(N−1) on the first superdiagonal. Shape (N, N) complex."""
    root_n = torch.sqrt(torch.arange(1, N, dtype=torch.float64, device=device))  # (N-1,)
    return torch.diag(root_n, 1).to(torch.complex128)


def position(a: torch.Tensor) -> torch.Tensor:
    """X = (a + a†)/√2. Shape (N, N), real symmetric (stored complex)."""
    return (a + a.conj().T) / math.sqrt(2)


def momentum(a: torch.Tensor) -> torch.Tensor:
    """P = i(a† − a)/√2. Shape (N, N), Hermitian with imaginary entries."""
    return 1j * (a.conj().T - a) / math.sqrt(2)


def coherent_state(alpha: complex, N: int, device: torch.device) -> torch.Tensor:
    """|α⟩: c_n = e^{−|α|²/2} αⁿ/√(n!), the minimum-uncertainty packet centred on x = √2 Re α. Shape (N,) complex.

    Computed through logarithms so αⁿ and n! never overflow; the tail beyond N is cut off.
    """
    n = torch.arange(N, dtype=torch.float64, device=device)  # (N,)
    if alpha == 0:
        return (n == 0).to(torch.complex128)  # |0⟩, the ground state
    log_mag = -(abs(alpha) ** 2) / 2 + n * math.log(abs(alpha)) - 0.5 * torch.lgamma(n + 1)
    phase = torch.exp(1j * n * math.atan2(alpha.imag, alpha.real))  # (N,)
    return torch.exp(log_mag) * phase
