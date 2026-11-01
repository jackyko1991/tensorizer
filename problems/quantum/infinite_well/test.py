"""Physics checks, written first (TDD): what any correct particle-in-a-box code must satisfy.

Run with `pytest problems/quantum/infinite_well/test.py`. Each test is a physical law, not an implementation detail:
the levels follow E_n ∝ n², eigenstates are orthonormal, probability is conserved and the state revives.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def well(n: int = 400, L: float = 1.5):
    x, dx = tensors.grid(n, L, CPU)
    energies, phi = solve.eigenstates(tensors.hamiltonian(n, L, CPU), dx)
    return x, dx, energies, phi


def test_energies_match_n_squared_law():
    L = 1.5
    _, _, energies, _ = well(L=L)
    n = torch.arange(1, 6, dtype=torch.float64)
    exact = n**2 * math.pi**2 / (2 * L**2)
    assert torch.allclose(energies[:5], exact, rtol=1e-3)  # finite-difference error ~ (nπ dx)²/12


def test_eigenvectors_are_orthonormal():
    _, dx, _, phi = well()
    gram = torch.einsum("xn,xm->nm", phi, phi) * dx  # ∫ φ_n φ_m dx
    # Hermitian H: eigenstates are orthonormal
    assert torch.allclose(gram, torch.eye(phi.shape[1], dtype=gram.dtype), atol=1e-10)


def test_norm_is_conserved_and_state_revives():
    x, dx, energies, phi = well()
    c = solve.coefficients(phi, tensors.gaussian_packet(x, 1.5, dx), dx)
    _, psi = solve.solve(phi, energies, c, t_max=0.7, n_steps=50)
    norms = equation.norm(psi, dx)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)  # unitary evolution: total probability stays 1

    # low states: grid energies ≈ n²E_1, so the initial state comes back after the revival time
    c = tensors.equal_superposition(3, len(x), CPU)
    _, psi = solve.solve(phi, energies, c, t_max=solve.revival_time(1.5), n_steps=10)
    overlap = (torch.einsum("x,x->", psi[0].conj(), psi[-1]) * dx).abs()
    assert overlap > 0.999  # E_n ∝ n²: every phase returns at the revival time
