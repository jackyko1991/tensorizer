"""Physics checks, written first (TDD): what any correct standing-wave code must satisfy.

Run with `pytest problems/waves/standing_waves/test.py`. Mode frequencies are harmonics f_n = n c / 2L, each harmonic
returns after one period, and energy is conserved at its analytic value.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_frequencies_are_harmonics():
    L, c = 2.0, 1.5
    x, dx = tensors.grid(L, 201, CPU)
    D = tensors.stencil(len(x) - 2, dx, CPU).double()
    f = equation.mode_frequencies(D, c)[:5]
    expected = torch.arange(1, 6, dtype=f.dtype) * c / (2 * L)
    assert torch.allclose(f, expected, rtol=1e-3)  # f_n = n c / 2L


def test_harmonic_returns_after_one_period_and_conserves_energy():
    L, c = 1.0, 1.0
    x, dx = tensors.grid(L, 101, CPU)
    D = tensors.stencil(len(x) - 2, dx, CPU)
    u0 = tensors.initial_shapes(x, L, n_harmonics=2, pluck_at=0.3)  # rows: n = 1, n = 2, plucked
    times, traj, dt = solve.solve(u0, D, c, dx, t_max=2 * L / c)  # one fundamental period
    assert torch.allclose(traj[-1, :2], u0[:2], atol=2e-2)  # harmonics are back
    # half a period of n = 1: the shape is upside down
    half = int(torch.argmin((times - L / c).abs()))
    assert torch.allclose(traj[half, 0], -u0[0], atol=2e-2)
    E = equation.energy(traj, dx, dt, c)  # (T-2, B)
    assert torch.allclose(E, E[0].expand_as(E), rtol=1e-2)  # no damping: energy is constant
    # analytic energy of sin(nπx/L) at rest: ½ c² (nπ/L)² · L/2
    n = torch.tensor([1.0, 2.0])
    assert torch.allclose(E[0, :2], 0.5 * c**2 * (n * math.pi / L) ** 2 * L / 2, rtol=2e-2)
