"""Physics checks, written first (TDD): what any correct Crank–Nicolson code must satisfy.

Run with `pytest problems/quantum/schrodinger_crank_nicolson/test.py`. Each test is a physical law, not an
implementation detail: the Cayley step is unitary so norm and energy are conserved, the error is second order in Δt,
explicit Euler is unstable and a harmonic packet follows the classical motion.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _setup(n: int = 200, g: float = 0.05, x0s=(2.0, 1.0)):
    x, dx = tensors.grid(n, 14.0, CPU)
    H = tensors.hamiltonian(x, dx, tensors.potential(x, g))
    return x, dx, H, tensors.packets(x, list(x0s))


def test_norm_is_conserved_every_step():
    _, dx, H, psi0 = _setup()
    A, B = equation.cayley_matrices(H, 0.01)
    lu, piv = torch.linalg.lu_factor(A)
    psi, n0 = psi0, equation.norm(psi0, dx)
    for _ in range(200):
        new = equation.cn_step(lu, piv, B, psi)
        assert torch.allclose(equation.norm(new, dx), equation.norm(psi, dx), rtol=0, atol=1e-12)  # per step
        psi = new
    assert torch.allclose(equation.norm(psi, dx), n0, rtol=0, atol=1e-11)


def test_step_matrix_is_unitary():
    _, _, H, _ = _setup(n=80)
    A, B = equation.cayley_matrices(H, 0.5)  # large Δt: still unitary
    U = torch.linalg.solve(A, B)
    eye = torch.eye(len(H), dtype=U.dtype)
    assert torch.allclose(U.conj().T @ U, eye, atol=1e-12)  # Cayley form (1 + iHΔt/2)⁻¹(1 − iHΔt/2) is unitary


def test_energy_is_conserved():
    _, dx, H, psi0 = _setup()
    _, psi = solve.crank_nicolson(H, psi0, 0.01, 5.0)
    E = equation.energy(psi, H, dx)  # (T, B)
    assert torch.allclose(E, E[0].expand_as(E), rtol=1e-11, atol=0)  # time-independent H: ⟨H⟩ is constant


def test_matches_exact_with_second_order_error():
    _, _, H, psi0 = _setup(n=150)
    t_end = 1.0
    exact = solve.exact(H, psi0, t_end)
    errors = []
    for dt in (0.01, 0.005):
        _, psi = solve.crank_nicolson(H, psi0, dt, t_end)
        errors.append((psi[-1] - exact).abs().max().item())
    assert errors[1] < 5e-3
    assert 3.6 < errors[0] / errors[1] < 4.4  # halving Δt quarters the error: second order


def test_explicit_euler_norm_grows():
    _, dx, H, psi0 = _setup()
    times, norms = solve.euler_norms(H, psi0, dx, 0.01, 5.0, cap=1e6)
    assert (norms[1:] >= norms[:-1] - 1e-12).all()  # never shrinks: |1 − iEΔt| > 1
    assert norms[-1].max() > 1e6


def test_lu_solve_equals_solve():
    _, _, H, psi0 = _setup(n=120)
    _, psi_lu = solve.crank_nicolson(H, psi0, 0.01, 0.2, use_lu=True)
    _, psi_solve = solve.crank_nicolson(H, psi0, 0.01, 0.2, use_lu=False)
    assert torch.allclose(psi_lu, psi_solve, atol=1e-12)


def test_harmonic_packet_follows_classical_motion():
    x, dx, H, psi0 = _setup(n=300, g=0.0, x0s=(2.0,))
    times, psi = solve.crank_nicolson(H, psi0, 0.005, math.pi)
    x_mean = equation.mean_position(psi, x, dx)[:, 0]
    assert torch.allclose(x_mean, 2.0 * torch.cos(times), atol=2e-2)  # Ehrenfest: ⟨x⟩ = x0 cos t in a harmonic well
