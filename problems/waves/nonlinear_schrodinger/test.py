"""Physics checks, written first (TDD): what any correct nonlinear-Schrödinger code must satisfy.

Run with `pytest problems/waves/nonlinear_schrodinger/test.py`. The bright soliton keeps its exact shape, norm and
Hamiltonian are conserved, and the Peregrine breather peaks at three times the background.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_bright_soliton_keeps_its_shape_and_invariants():
    eta, v, dt, t_max = 1.5, tensors.SOLITON_SPEED, 0.005, 10.0
    x, k, dx = tensors.grid(512, CPU)
    psi0 = tensors.initial_wave(x, "soliton", 0.0, eta)
    times, psi = solve.solve(psi0, tensors.half_linear_step(k, dt), dt, t_max, n_frames=11)
    exact = equation.bright_soliton(x + 10.0, t_max, eta, v)
    assert torch.allclose(psi[-1].abs(), exact.abs(), atol=1e-3)
    N, H = equation.norm(psi, dx), equation.hamiltonian(psi, k, dx)
    assert N[0].item() == pytest.approx(2 * eta, rel=1e-6)  # ∫ η² sech² = 2η
    assert torch.allclose(N, N[0].expand_as(N), rtol=1e-10)
    assert torch.allclose(H, H[0].expand_as(H), rtol=1e-6)


def test_modulational_instability_and_peregrine_peak():
    dt = 0.005
    x, k, dx = tensors.grid(512, CPU)
    half = tensors.half_linear_step(k, dt)
    # Peregrine breather started at t = -5 peaks at |psi| = 3 at t = 5
    times, psi = solve.solve(tensors.initial_wave(x, "peregrine", 0.0, 1.0), half, dt, 7.0, n_frames=71)
    peak = psi.abs().max(dim=1).values
    assert peak.max().item() == pytest.approx(3.0, abs=0.02)
    assert times[peak.argmax()].item() == pytest.approx(5.0, abs=0.15)
    # a noisy plane wave grows peaks well above the background while N and H stay put
    times, psi = solve.solve(tensors.initial_wave(x, "modulation", 1e-3, 1.0), half, dt, 20.0, n_frames=41)
    assert psi[0].abs().max() < 1.01
    assert psi.abs().max() > 2.5
    N, H = equation.norm(psi, dx), equation.hamiltonian(psi, k, dx)
    assert torch.allclose(N, N[0].expand_as(N), rtol=1e-10)
    assert (H - H[0]).abs().max().item() < 1e-3 * N[0].item()
