"""Physics checks, written first (TDD): what any correct coupled-oscillator code must satisfy.

Run with `pytest problems/mechanics/coupled_oscillators/test.py`. Normal-mode frequencies match the analytic chain
result, modes are orthonormal, energy is conserved and the mode superposition equals direct integration of Newton's
second law.
"""

import torch

from problems._common.integrators import integrate

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_frequencies_match_analytic():
    for n in (2, 5, 10):
        K = tensors.stiffness(n, k=4.0, device=CPU)
        omega, e = equation.normal_modes(K, m=0.5)
        assert torch.allclose(omega, tensors.analytic_frequencies(n, k=4.0, m=0.5), rtol=1e-5)
        assert torch.allclose(e.T @ e, torch.eye(n), atol=1e-5)  # orthonormal modes


def test_energy_is_conserved():
    K = tensors.stiffness(6, k=10.0, device=CPU)
    x0 = tensors.initial_displacement(6, "first mass", 0.3, CPU)
    sol = solve.solve(K, 1.0, x0, t_max=20.0)
    kinetic, potential = equation.energy(sol["x"], sol["v"], K, 1.0)
    total = kinetic + potential
    assert torch.allclose(total, total[0].expand_as(total), rtol=1e-5)
    assert torch.allclose(total[0], potential[0])  # starts at rest


def test_modes_match_direct_integration():
    """The normal-mode superposition equals a brute-force RK4 solution of m ẍ_i = -K_ij x_j."""
    K, m = tensors.stiffness(4, k=3.0, device=CPU), 2.0
    x0 = tensors.initial_displacement(4, "middle mass", 0.2, CPU)
    sol = solve.solve(K, m, x0, t_max=5.0, n_times=501)  # dt = 0.01

    def rhs(y, t):
        return torch.stack([y[1], equation.spring_force(K, y[0]) / m])

    traj = integrate(rhs, torch.stack([x0, torch.zeros_like(x0)]), 0.01, 500)
    assert torch.allclose(traj[:, 0], sol["x"], atol=1e-4)


def test_single_mode_keeps_its_shape():
    K = tensors.stiffness(5, k=1.0, device=CPU)
    x0 = tensors.initial_displacement(5, "mode 2", 0.3, CPU)
    sol = solve.solve(K, 1.0, x0, t_max=10.0)
    assert (sol["A"] > 1e-4).sum() == 1  # only one mode is excited
    ratio = sol["x"][:, 0] / x0[0]  # every mass follows the same cos(ω t)
    assert torch.allclose(sol["x"], ratio[:, None] * x0, atol=1e-5)
