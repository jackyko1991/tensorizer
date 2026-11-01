"""Physics checks, written first (TDD): what any correct quantum harmonic oscillator code must satisfy.

Run with `pytest problems/quantum/harmonic_oscillator/test.py`. Each test is a physical law, not an implementation
detail: the levels are ħω(n + ½), probability is conserved and a coherent state follows the classical motion.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def oscillator(omega: float = 1.3, n: int = 600, L: float = 20.0):
    x, dx = tensors.grid(n, L, CPU)
    energies, phi = solve.eigenstates(tensors.hamiltonian(x, dx, omega), dx)
    return x, dx, energies, phi


def test_energy_ladder_is_n_plus_half():
    omega = 1.3
    _, _, energies, _ = oscillator(omega)
    n = torch.arange(8, dtype=torch.float64)
    assert torch.allclose(energies[:8], omega * (n + 0.5), rtol=2e-3)  # E_n = ħω(n + ½)


def test_coherent_state_follows_classical_motion():
    omega, x0 = 1.3, 2.5
    x, dx, energies, phi = oscillator(omega)
    c = solve.coefficients(phi, tensors.coherent_state(x, x0, omega), dx)
    times, psi = solve.solve(phi, energies, c, t_max=2 * 2 * math.pi / omega, n_steps=80)
    x_mean = equation.mean_position(psi, x, dx)
    # ~1% phase lag after two periods: finite differences make the rungs slightly narrower than ω
    assert torch.allclose(x_mean, x0 * torch.cos(omega * times), atol=0.03)
    # unitary evolution: probability conserved
    assert torch.allclose(equation.norm(psi, dx), torch.ones(len(times), dtype=torch.float64), atol=1e-8)
