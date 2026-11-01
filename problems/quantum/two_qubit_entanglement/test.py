"""Physics checks, written first (TDD): what any correct two-qubit code must satisfy.

Run with `pytest problems/quantum/two_qubit_entanglement/test.py`. Each test is a physical law, not an
implementation detail: a Bell state has entropy ln 2, product states have none, gates are unitary and Ising
evolution conserves the norm while entangling the qubits.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_bell_state_entropy_and_reduced_state():
    bell = solve.bell_circuit(CPU)
    assert torch.allclose(bell, torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128) / math.sqrt(2))
    assert abs(equation.entanglement_entropy(bell) - math.log(2)) < 1e-12  # maximally entangled: S = ln 2
    assert abs(equation.concurrence(bell) - 1) < 1e-12
    half_identity = torch.eye(2, dtype=torch.complex128) / 2
    assert torch.allclose(equation.reduced_density_1(bell), half_identity)
    assert torch.allclose(equation.reduced_density_2(bell), half_identity)


def test_product_state_partial_trace_and_zero_entropy():
    a = torch.tensor([0.6, 0.8j], dtype=torch.complex128)
    b = torch.tensor([1, 1 - 1j], dtype=torch.complex128) / math.sqrt(3)
    psi = torch.einsum("a,b->ab", a, b)  # |a⟩ ⊗ |b⟩
    assert equation.entanglement_entropy(psi) < 1e-10  # product state: no entanglement
    assert equation.concurrence(psi) < 1e-12
    assert torch.allclose(equation.reduced_density_1(psi), torch.outer(a, a.conj()))
    assert torch.allclose(equation.reduced_density_2(psi), torch.outer(b, b.conj()))


def test_gate_tensors_unitary_and_hamiltonian_matches_kron():
    sigma = tensors.pauli(CPU)
    eye4 = torch.eye(4, dtype=torch.complex128)
    for G in [tensors.cnot(CPU), tensors.on_qubit_1(tensors.hadamard(CPU)), tensors.on_qubit_2(sigma[1])]:
        M = G.reshape(4, 4)
        assert torch.allclose(M.conj().T @ M, eye4)  # gates are unitary: U†U = 1
    J, h = 0.7, 0.3
    X, Z, eye2 = sigma[0], sigma[2], torch.eye(2, dtype=torch.complex128)
    kron = J * torch.kron(Z, Z) + h * (torch.kron(X, eye2) + torch.kron(eye2, X))
    assert torch.allclose(tensors.ising_hamiltonian(J, h, sigma).reshape(4, 4), kron)


def test_ising_evolution_unitary_and_entangles_product_state():
    sigma = tensors.pauli(CPU)
    H = tensors.ising_hamiltonian(1.0, 0.6, sigma)
    result = solve.solve(tensors.basis_state(0, 0, CPU), H, sigma, t_max=5.0, n_steps=100)
    norms = torch.einsum("tab,tab->t", result["psi"].conj(), result["psi"]).real
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)  # unitary evolution conserves the norm
    assert result["entropy"][0] < 1e-10 and result["entropy"].max() > 0.3
    # a pure global state has equally mixed halves: |r1| = |r2|
    assert torch.allclose(result["r1"].norm(dim=1), result["r2"].norm(dim=1), atol=1e-10)
