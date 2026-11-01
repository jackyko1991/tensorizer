"""Run the Bell circuit, then evolve the pair under the Ising Hamiltonian and measure entanglement."""

import torch

from . import tensors
from .equation import (
    apply_gate,
    bloch_vector,
    concurrence,
    entanglement_entropy,
    evolve,
    propagators,
    reduced_density_1,
    reduced_density_2,
)


def bell_circuit(device: torch.device) -> torch.Tensor:
    """|00⟩ → (H ⊗ I) → CNOT → (|00⟩ + |11⟩)/√2. Returns ψ_ab, shape (2, 2)."""
    psi = tensors.basis_state(0, 0, device)
    psi = apply_gate(tensors.on_qubit_1(tensors.hadamard(device)), psi)  # (|0⟩ + |1⟩)|0⟩ / √2
    return apply_gate(tensors.cnot(device), psi)  # flip qubit 2 when qubit 1 is |1⟩


def solve(psi0: torch.Tensor, H: torch.Tensor, sigma: torch.Tensor, t_max: float, n_steps: int = 240) -> dict:
    """Exact evolution at T = n_steps + 1 times; every quantity is computed for all times in one batch.

    Returns a dict of tensors: times (T,), psi (T, 2, 2), entropy (T,), concurrence (T,), r1 and r2 (T, 3).
    """
    times = torch.linspace(0.0, t_max, n_steps + 1, dtype=torch.float64, device=psi0.device)  # (T,)
    psi = evolve(propagators(H, times), psi0)  # (T, 2, 2)
    return {
        "times": times,
        "psi": psi,
        "entropy": entanglement_entropy(psi),
        "concurrence": concurrence(psi),
        "r1": bloch_vector(reduced_density_1(psi), sigma),
        "r2": bloch_vector(reduced_density_2(psi), sigma),
    }
