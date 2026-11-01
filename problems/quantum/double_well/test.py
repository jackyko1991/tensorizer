"""Physics checks, written first (TDD): what any correct double-well code must satisfy.

Run with `pytest problems/quantum/double_well/test.py`. Each test is a physical law, not an implementation detail:
the lowest pair has even and odd parity, and the packet tunnels back and forth with period 2π/ΔE as the two-level
model predicts.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def double_well(V0: float = 3.0, a: float = 1.5, n: int = 400):
    x, dx = tensors.grid(n, 4 * a + 6, CPU)
    energies, phi = solve.eigenstates(tensors.hamiltonian(x, dx, tensors.double_well(x, V0, a)), x, dx)
    return x, dx, energies, phi


def test_doublet_has_definite_parity():
    _, _, energies, phi = double_well()
    assert energies[1] - energies[0] < 0.1 * (energies[2] - energies[1])  # a close pair below the rest
    assert torch.allclose(phi[:, 0].flip(0), phi[:, 0], atol=1e-8)  # even
    assert torch.allclose(phi[:, 1].flip(0), -phi[:, 1], atol=1e-8)  # odd


def test_tunnelling_period_and_cos2():
    x, dx, energies, phi = double_well()
    delta_E = (energies[1] - energies[0]).item()
    period = solve.tunnelling_period(energies)
    times, psi = solve.solve(phi, energies, tensors.left_state(len(x), CPU), t_max=period, n_steps=60)
    p_left = equation.left_probability(psi, x, dx)
    # two-level model: P_L = cos²(ΔE t / 2)
    assert torch.allclose(p_left, equation.two_state_prediction(times, delta_E), atol=0.03)
    assert p_left[0] > 0.97 and p_left[30] < 0.03  # left at t = 0, right at half a period
    assert abs(p_left[-1] - p_left[0]) < 1e-8  # back exactly after 2π/ΔE
