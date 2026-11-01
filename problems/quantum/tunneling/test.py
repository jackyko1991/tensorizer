"""Physics checks, written first (TDD): what any correct tunnelling code must satisfy.

Run with `pytest problems/quantum/tunneling/test.py`. Each test is a physical law, not an implementation detail:
probability is conserved, a free packet moves at the group velocity and spreads exactly, and a wide packet tunnels
with the plane-wave transmission coefficient.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_norm_is_conserved():
    x, k, dx = tensors.grid(1024, 200.0, CPU)
    V = tensors.barrier(x, 2.5, 1.0)
    psi0 = tensors.gaussian_packet(x, -25.0, 2.0, 2.0, dx)
    _, psi = solve.solve(psi0, V, k, t_max=10.0)
    norms = equation.probability(psi, dx)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)  # unitary evolution: total probability stays 1


def test_free_packet_moves_at_k0_and_spreads():
    x, k, dx = tensors.grid(2048, 200.0, CPU)
    x0, sigma0, k0, t = -30.0, 2.0, 3.0, 15.0
    psi0 = tensors.gaussian_packet(x, x0, sigma0, k0, dx)
    times, psi = solve.solve(psi0, torch.zeros_like(x), k, t_max=t)
    rho = psi[-1].abs() ** 2 * dx  # (N,) probability per cell
    mean = (rho * x).sum()
    width = torch.sqrt((rho * (x - mean) ** 2).sum())
    assert abs(mean - (x0 + k0 * times[-1])) < 1e-3  # group velocity = k0 (ħ = m = 1)
    expected = sigma0 * math.sqrt(1 + (times[-1] / (2 * sigma0**2)) ** 2)
    assert abs(width - expected) / expected < 1e-3  # exact spreading of a free Gaussian


def test_wide_packet_transmission_matches_plane_wave():
    # σ0 = 12 ⇒ momentum spread 1/(2σ0) ≈ 0.04, nearly a plane wave of k0 = 2
    x, k, dx = tensors.grid(8192, 600.0, CPU)
    k0, V0, a = 2.0, 2.5, 1.0
    psi0 = tensors.gaussian_packet(x, -100.0, 12.0, k0, dx)
    _, psi = solve.solve(psi0, tensors.barrier(x, V0, a), k, t_max=100.0, dt=0.02, n_saved=2)
    p_trans = solve.transmission(psi[-1], x, a, dx)
    expected = solve.plane_wave_transmission(k0**2 / 2, V0, tensors.grid_width(x, a, dx))
    assert abs(p_trans - expected) < 0.01  # analytic plane-wave transmission T(E)
