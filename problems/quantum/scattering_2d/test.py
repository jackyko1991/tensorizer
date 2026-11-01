"""Physics checks, written first (TDD): what any correct 2D scattering code must satisfy.

Run with `pytest problems/quantum/scattering_2d/test.py`. Each test is a physical law, not an implementation detail:
probability is conserved without an absorber, a free packet moves at the group velocity and the split-step
propagator is unitary.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_norm_is_conserved_without_absorber():
    X, Y, KX, KY, dx = tensors.grid(128, 60.0, CPU)
    V = tensors.double_slit(X, Y, 40.0, 6.0, 1.5)
    psi0 = tensors.gaussian_packet(X, Y, -12.0, 3.0, 4.0, 3.0, dx * dx)
    result = solve.solve(psi0, V, KX, KY, dx * dx, t_max=4.0, M=None, n_saved=5)
    # unitary evolution: total probability stays 1
    assert torch.allclose(result["norm"], torch.ones_like(result["norm"]), atol=1e-10)


def test_free_packet_moves_at_k0():
    X, Y, KX, KY, dx = tensors.grid(128, 60.0, CPU)
    x0, k0, t = -10.0, 2.5, 6.0
    psi0 = tensors.gaussian_packet(X, Y, x0, 3.0, 3.0, k0, dx * dx)
    result = solve.solve(psi0, torch.zeros_like(X), KX, KY, dx * dx, t_max=t, n_saved=2)
    centre = solve.mean_position(result["psi"].abs() ** 2, X, Y)
    assert abs(centre[0] - (x0 + k0 * t)) < 1e-3  # Ehrenfest: ⟨x⟩ = x0 + k0 t
    assert abs(centre[1]) < 1e-6


def test_kinetic_operator_diagonal_in_k_and_step_unitary():
    X, Y, KX, KY, dx = tensors.grid(32, 20.0, CPU)
    # a plane wave on the grid is a single nonzero entry of fft2: the Fourier basis diagonalises −½∇²
    plane = torch.exp(1j * (KX[3, 0] * X + KY[0, 5] * Y))
    spectrum = torch.fft.fft2(plane).abs()
    assert (spectrum > 1e-8).sum() == 1 and spectrum[3, 5] > 1
    K = equation.kinetic_phase(KX, KY, 0.05)
    assert torch.allclose(K.abs(), torch.ones_like(K.abs()))
    # one full split step (no absorber) preserves the norm of an arbitrary state
    gen = torch.Generator().manual_seed(0)
    psi = torch.randn(32, 32, dtype=torch.complex128, generator=gen)
    P = equation.potential_phase(tensors.disk(X, Y, 5.0, 4.0), 0.05)
    after = equation.split_step(psi, P, K, torch.ones_like(X))
    # each split step is unitary
    assert torch.allclose(equation.probability(after, 1.0), equation.probability(psi, 1.0))
