"""Physics checks for the two formulation problems: Crank–Nicolson time stepping and matrix mechanics."""

import math

import pytest
import torch

from problems.quantum.matrix_mechanics import equation as mm_eq
from problems.quantum.matrix_mechanics import solve as mm_solve
from problems.quantum.matrix_mechanics import tensors as mm_t
from problems.quantum.schrodinger_crank_nicolson import equation as cn_eq
from problems.quantum.schrodinger_crank_nicolson import solve as cn_solve
from problems.quantum.schrodinger_crank_nicolson import tensors as cn_t

CPU = torch.device("cpu")


# --- Crank–Nicolson ---------------------------------------------------------------------------------------


def _cn_setup(n: int = 200, g: float = 0.05, x0s=(2.0, 1.0)):
    x, dx = cn_t.grid(n, 14.0, CPU)
    H = cn_t.hamiltonian(x, dx, cn_t.potential(x, g))
    return x, dx, H, cn_t.packets(x, list(x0s))


def test_cn_norm_conserved_every_step():
    _, dx, H, psi0 = _cn_setup()
    A, B = cn_eq.cayley_matrices(H, 0.01)
    lu, piv = torch.linalg.lu_factor(A)
    psi, n0 = psi0, cn_eq.norm(psi0, dx)
    for _ in range(200):
        new = cn_eq.cn_step(lu, piv, B, psi)
        assert torch.allclose(cn_eq.norm(new, dx), cn_eq.norm(psi, dx), rtol=0, atol=1e-12)  # per step
        psi = new
    assert torch.allclose(cn_eq.norm(psi, dx), n0, rtol=0, atol=1e-11)


def test_cn_step_matrix_is_unitary():
    _, _, H, _ = _cn_setup(n=80)
    A, B = cn_eq.cayley_matrices(H, 0.5)  # large Δt: still unitary
    U = torch.linalg.solve(A, B)
    eye = torch.eye(len(H), dtype=U.dtype)
    assert torch.allclose(U.conj().T @ U, eye, atol=1e-12)


def test_cn_energy_conserved():
    _, dx, H, psi0 = _cn_setup()
    _, psi = cn_solve.crank_nicolson(H, psi0, 0.01, 5.0)
    E = cn_eq.energy(psi, H, dx)  # (T, B)
    assert torch.allclose(E, E[0].expand_as(E), rtol=1e-11, atol=0)


def test_cn_matches_exact_with_second_order_error():
    _, _, H, psi0 = _cn_setup(n=150)
    t_end = 1.0
    exact = cn_solve.exact(H, psi0, t_end)
    errors = []
    for dt in (0.01, 0.005):
        _, psi = cn_solve.crank_nicolson(H, psi0, dt, t_end)
        errors.append((psi[-1] - exact).abs().max().item())
    assert errors[1] < 5e-3
    assert 3.6 < errors[0] / errors[1] < 4.4  # halving Δt quarters the error: second order


def test_explicit_euler_norm_grows():
    _, dx, H, psi0 = _cn_setup()
    times, norms = cn_solve.euler_norms(H, psi0, dx, 0.01, 5.0, cap=1e6)
    assert (norms[1:] >= norms[:-1] - 1e-12).all()  # never shrinks: |1 − iEΔt| > 1
    assert norms[-1].max() > 1e6


def test_lu_solve_equals_solve():
    _, _, H, psi0 = _cn_setup(n=120)
    _, psi_lu = cn_solve.crank_nicolson(H, psi0, 0.01, 0.2, use_lu=True)
    _, psi_solve = cn_solve.crank_nicolson(H, psi0, 0.01, 0.2, use_lu=False)
    assert torch.allclose(psi_lu, psi_solve, atol=1e-12)


def test_cn_harmonic_packet_follows_classical_motion():
    x, dx, H, psi0 = _cn_setup(n=300, g=0.0, x0s=(2.0,))
    times, psi = cn_solve.crank_nicolson(H, psi0, 0.005, math.pi)
    x_mean = cn_eq.mean_position(psi, x, dx)[:, 0]
    assert torch.allclose(x_mean, 2.0 * torch.cos(times), atol=2e-2)


# --- matrix mechanics -------------------------------------------------------------------------------------


def _mm(N: int = 30):
    a = mm_t.lowering(N, CPU)
    return a, mm_t.position(a), mm_t.momentum(a)


def test_mm_matrices_hermitian():
    a, X, P = _mm()
    for M in (X, P, mm_eq.hamiltonian(X, P), mm_eq.hamiltonian(X, P, 0.05)):
        assert torch.allclose(M, mm_eq.dagger(M))
    assert not torch.allclose(a, mm_eq.dagger(a))


def test_mm_eigenvalues_n_plus_half():
    N = 30
    a, X, P = _mm(N)
    H = mm_eq.hamiltonian(X, P)
    # ½(P² + X²) = a†a + ½ except its last diagonal entry, which truncation sets to (N − 1)/2
    assert torch.allclose(H[:-1, :-1], mm_eq.number_hamiltonian(a)[:-1, :-1], atol=1e-12)
    assert H[-1, -1].real.item() == pytest.approx((N - 1) / 2)
    expected = sorted([n + 0.5 for n in range(N - 1)] + [(N - 1) / 2])
    E = mm_solve.energies(H)
    assert torch.allclose(E, torch.tensor(expected, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(E[: N // 2], torch.arange(N // 2, dtype=torch.float64) + 0.5, atol=1e-10)


def test_mm_commutator():
    N = 25
    _, X, P = _mm(N)
    C = mm_eq.commutator(X, P)
    expected = 1j * torch.eye(N, dtype=torch.complex128)
    expected[-1, -1] = 1j * (1 - N)
    assert torch.allclose(C, expected, atol=1e-12)
    assert abs(torch.trace(C).item()) < 1e-10  # a commutator of finite matrices is traceless


def test_mm_pictures_agree_and_coherent_state_is_classical():
    N, alpha = 40, 1.7
    _, X, P = _mm(N)
    H = mm_eq.hamiltonian(X, P)
    times = torch.linspace(0, 4 * math.pi, 50, dtype=torch.float64)
    U = mm_solve.propagators(H, times)
    c = mm_t.coherent_state(complex(alpha, 0.0), N, CPU)
    assert torch.linalg.vector_norm(c).item() == pytest.approx(1.0, abs=1e-12)
    heis, _ = mm_solve.heisenberg_picture(U, {"X": X, "P": P}, c)
    schr = mm_solve.schrodinger_picture(U, {"X": X, "P": P}, c)
    assert torch.allclose(heis["X"], schr["X"], atol=1e-12)
    assert torch.allclose(heis["P"], schr["P"], atol=1e-12)
    phase = alpha * torch.exp(-1j * times)
    assert torch.allclose(heis["X"], math.sqrt(2) * phase.real, atol=1e-9)
    assert torch.allclose(heis["P"], math.sqrt(2) * phase.imag, atol=1e-9)


def test_mm_heisenberg_operator_is_classical_solution():
    _, X, P = _mm(20)
    H = mm_eq.number_hamiltonian(mm_t.lowering(20, CPU))  # exact diagonal H: no truncation artefact
    t = torch.tensor([0.7], dtype=torch.float64)
    X_t = mm_eq.heisenberg(mm_solve.propagators(H, t), X)[0]
    assert torch.allclose(X_t, X * math.cos(0.7) + P * math.sin(0.7), atol=1e-12)


def test_mm_anharmonic_matches_first_order_perturbation():
    N = 40
    _, X, P = _mm(N)
    n = torch.arange(6, dtype=torch.float64)
    deviations = []
    for lam in (2e-4, 1e-4):
        E = mm_solve.energies(mm_eq.hamiltonian(X, P, lam))[:6]
        pert = mm_solve.first_order_energies(n, lam)
        assert torch.allclose(E, pert, atol=2e-5)
        deviations.append((E - pert).abs().max().item())
    assert 3.5 < deviations[0] / deviations[1] < 4.5  # what's left is second order in λ
