"""Physics checks, written first (TDD): what any correct KdV code must satisfy.

Run with `pytest problems/waves/kdv_solitons/test.py`. A soliton of height c/2 moves at speed c without changing shape,
and collisions conserve mass and momentum and shift the solitons by the analytic phase.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_single_soliton_speed_is_twice_its_height():
    c = 2.0
    x, k, dx = tensors.grid(512, CPU)
    u0 = equation.soliton(x, c, 30.0)
    assert equation.soliton(torch.tensor([30.0], dtype=torch.float64), c, 30.0).item() == pytest.approx(c / 2)
    times, u = solve.solve(u0, k, tensors.dealias_mask(k), t_max=10.0, n_frames=11)
    positions = torch.tensor([solve.peak_position(x, ui) for ui in u], dtype=torch.float64)
    speed = ((positions[-1] - positions[0]) / (times[-1] - times[0])).item()
    assert speed == pytest.approx(c, rel=5e-3)
    assert u[-1].max().item() == pytest.approx(u0.max().item(), rel=5e-3)  # keeps its height


def test_collision_conserves_mass_momentum_and_shifts_solitons():
    c1, c2, t_max = 4.0, 1.0, 12.0
    x, k, dx = tensors.grid(1024, CPU)
    u0 = tensors.two_solitons(x, c1, c2)
    times, u = solve.solve(u0, k, tensors.dealias_mask(k), t_max)
    M, P = equation.mass(u, dx), equation.momentum(u, dx)
    assert torch.allclose(M, M[0].expand_as(M), rtol=1e-10)
    assert torch.allclose(P, P[0].expand_as(P), rtol=1e-3)
    x1 = solve.peak_position(x, u[-1])
    x2 = solve.second_peak_position(x, u[-1], x1)
    d1, d2 = equation.phase_shifts(c1, c2)
    assert x1 - (tensors.X_TALL + c1 * t_max) == pytest.approx(d1, abs=0.05)  # tall soliton jumps ahead
    assert x2 - (tensors.X_SHORT + c2 * t_max) == pytest.approx(-d2, abs=0.05)  # short one falls behind
