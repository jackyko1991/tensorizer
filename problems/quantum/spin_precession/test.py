"""Physics checks, written first (TDD): what any correct spin-precession code must satisfy.

Run with `pytest problems/quantum/spin_precession/test.py`. Each test is a physical law, not an implementation
detail: the Pauli algebra holds, the spin precesses at the Larmor frequency, the Bloch vector stays on the unit
sphere and a resonant drive flips the spin after half a Rabi period.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_pauli_algebra():
    sigma = tensors.pauli(CPU)
    eps = tensors.levi_civita(CPU)
    lhs = torch.einsum("iab,jbc->ijac", sigma, sigma)  # σ_i σ_j
    eye = torch.eye(2, dtype=sigma.dtype)
    delta = torch.eye(3, dtype=sigma.dtype)
    rhs = torch.einsum("ij,ac->ijac", delta, eye) + 1j * torch.einsum("ijk,kac->ijac", eps, sigma)
    assert torch.allclose(lhs, rhs)  # σ_i σ_j = δ_ij + i ε_ijk σ_k


def test_larmor_precession_frequency():
    sigma = tensors.pauli(CPU)
    gamma, B0 = 1.7, 2.0
    psi0 = tensors.tilted_spinor(90.0, CPU)  # along +x
    times, _, s = solve.solve(psi0, sigma, gamma, B0, 0.0, 0.0, t_max=5.0)
    w = gamma * B0
    # ds/dt = γ s × B with B along z: s rotates clockwise seen from +z
    expected = torch.stack([torch.cos(w * times), -torch.sin(w * times), torch.zeros_like(times)], dim=1)
    assert torch.allclose(s, expected, atol=1e-8)  # Larmor precession at ω = γB


def test_bloch_vector_stays_unit_with_drive():
    sigma = tensors.pauli(CPU)
    psi0 = tensors.tilted_spinor(30.0, CPU)
    _, psi, s = solve.solve(psi0, sigma, 1.0, 1.0, 0.4, 1.0, t_max=20.0)
    # pure state under unitary evolution: |s| = 1
    assert torch.allclose(s.norm(dim=1), torch.ones(len(s), dtype=s.dtype), atol=1e-10)
    assert torch.allclose(equation.bloch_vector(psi[0], sigma), torch.tensor([0.5, 0.0, math.sqrt(3) / 2]).double())


def test_resonant_drive_flips_spin_at_rabi_frequency():
    sigma = tensors.pauli(CPU)
    gamma, B0, B1 = 1.0, 1.0, 0.05
    t_flip = math.pi / (gamma * B1 / 2)  # half a Rabi period
    psi0 = tensors.tilted_spinor(0.0, CPU)
    _, _, s = solve.solve(psi0, sigma, gamma, B0, B1, gamma * B0, t_max=t_flip)
    assert s[-1, 2] < -0.99  # Rabi: full flip after half a Rabi period
