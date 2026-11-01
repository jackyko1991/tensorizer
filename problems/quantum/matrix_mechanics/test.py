"""Physics checks, written first (TDD): what any correct matrix-mechanics code must satisfy.

Run with `pytest problems/quantum/matrix_mechanics/test.py`. Each test is a physical law, not an implementation
detail: observables are Hermitian, the spectrum is n + ½, [X, P] = i (up to truncation), the Heisenberg and
Schrödinger pictures agree and the anharmonic shift matches perturbation theory.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _mm(N: int = 30):
    a = tensors.lowering(N, CPU)
    return a, tensors.position(a), tensors.momentum(a)


def test_matrices_are_hermitian():
    a, X, P = _mm()
    for M in (X, P, equation.hamiltonian(X, P), equation.hamiltonian(X, P, 0.05)):
        assert torch.allclose(M, equation.dagger(M))  # observables are Hermitian
    assert not torch.allclose(a, equation.dagger(a))


def test_eigenvalues_are_n_plus_half():
    N = 30
    a, X, P = _mm(N)
    H = equation.hamiltonian(X, P)
    # ½(P² + X²) = a†a + ½ except its last diagonal entry, which truncation sets to (N − 1)/2
    assert torch.allclose(H[:-1, :-1], equation.number_hamiltonian(a)[:-1, :-1], atol=1e-12)
    assert H[-1, -1].real.item() == pytest.approx((N - 1) / 2)
    expected = sorted([n + 0.5 for n in range(N - 1)] + [(N - 1) / 2])
    E = solve.energies(H)
    assert torch.allclose(E, torch.tensor(expected, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(E[: N // 2], torch.arange(N // 2, dtype=torch.float64) + 0.5, atol=1e-10)


def test_canonical_commutator():
    N = 25
    _, X, P = _mm(N)
    C = equation.commutator(X, P)
    expected = 1j * torch.eye(N, dtype=torch.complex128)
    expected[-1, -1] = 1j * (1 - N)
    assert torch.allclose(C, expected, atol=1e-12)  # [X, P] = iħ except the truncation corner
    assert abs(torch.trace(C).item()) < 1e-10  # a commutator of finite matrices is traceless


def test_pictures_agree_and_coherent_state_is_classical():
    N, alpha = 40, 1.7
    _, X, P = _mm(N)
    H = equation.hamiltonian(X, P)
    times = torch.linspace(0, 4 * math.pi, 50, dtype=torch.float64)
    U = solve.propagators(H, times)
    c = tensors.coherent_state(complex(alpha, 0.0), N, CPU)
    assert torch.linalg.vector_norm(c).item() == pytest.approx(1.0, abs=1e-12)
    heis, _ = solve.heisenberg_picture(U, {"X": X, "P": P}, c)
    schr = solve.schrodinger_picture(U, {"X": X, "P": P}, c)
    assert torch.allclose(heis["X"], schr["X"], atol=1e-12)  # Heisenberg and Schrödinger pictures agree
    assert torch.allclose(heis["P"], schr["P"], atol=1e-12)
    phase = alpha * torch.exp(-1j * times)
    assert torch.allclose(heis["X"], math.sqrt(2) * phase.real, atol=1e-9)
    assert torch.allclose(heis["P"], math.sqrt(2) * phase.imag, atol=1e-9)


def test_heisenberg_operator_is_classical_solution():
    _, X, P = _mm(20)
    H = equation.number_hamiltonian(tensors.lowering(20, CPU))  # exact diagonal H: no truncation artefact
    t = torch.tensor([0.7], dtype=torch.float64)
    X_t = equation.heisenberg(solve.propagators(H, t), X)[0]
    # operator obeys the classical equation of motion
    assert torch.allclose(X_t, X * math.cos(0.7) + P * math.sin(0.7), atol=1e-12)


def test_anharmonic_matches_first_order_perturbation():
    N = 40
    _, X, P = _mm(N)
    n = torch.arange(6, dtype=torch.float64)
    deviations = []
    for lam in (2e-4, 1e-4):
        E = solve.energies(equation.hamiltonian(X, P, lam))[:6]
        pert = solve.first_order_energies(n, lam)
        assert torch.allclose(E, pert, atol=2e-5)
        deviations.append((E - pert).abs().max().item())
    assert 3.5 < deviations[0] / deviations[1] < 4.5  # what's left is second order in λ
