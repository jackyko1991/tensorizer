"""Physics checks, written first (TDD): what any correct pendulum code must satisfy.

Run with `pytest problems/mechanics/simple_pendulum/test.py`. Small swings have T = 2π√(L/g), large swings follow the
exact elliptic-integral period, and energy is conserved.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_small_amplitude_period():
    g, L = 9.81, 1.5
    y0 = tensors.initial_state(torch.deg2rad(torch.tensor([2.0])), CPU)
    times, traj = solve.solve(y0, g, L, t_max=3.0, dt=0.001)
    period = solve.periods(times, traj[:, :, 0])
    assert period.item() == pytest.approx(2 * math.pi * math.sqrt(L / g), rel=1e-3)  # small-angle limit T₀ = 2π√(L/g)


def test_large_amplitude_is_slower_and_matches_elliptic_integral():
    g, L = 9.81, 1.0
    theta0 = torch.deg2rad(torch.tensor([30.0, 90.0, 150.0]))
    times, traj = solve.solve(tensors.initial_state(theta0, CPU), g, L, t_max=4.0, dt=0.001)
    measured = solve.periods(times, traj[:, :, 0])
    small = 2 * math.pi * math.sqrt(L / g)
    assert (measured > small).all() and (measured.diff() > 0).all()
    assert torch.allclose(measured, equation.exact_period(theta0, g, L), rtol=2e-3)  # exact period 4√(L/g) K(sin θ₀/2)
    # series check at 90°: T ≈ T0 (1 + θ²/16 + 11θ⁴/3072)
    series = small * (1 + (math.pi / 2) ** 2 / 16 + 11 * (math.pi / 2) ** 4 / 3072)
    assert measured[1].item() == pytest.approx(series, rel=0.01)


def test_energy_is_conserved():
    y0 = tensors.initial_state(tensors.amplitudes(4, 160.0), CPU)
    _, traj = solve.solve(y0, 9.81, 1.0, t_max=6.0)
    e = equation.energy(traj, 9.81, 1.0)  # (T, B)
    assert torch.allclose(e, e[0].expand_as(e), rtol=1e-4, atol=1e-5)  # no friction: total energy is constant


def test_small_angle_solution_matches_for_tiny_swing():
    y0 = tensors.initial_state(torch.deg2rad(torch.tensor([1.0])), CPU)
    times, traj = solve.solve(y0, 9.81, 1.0, t_max=4.0)
    small = equation.small_angle(y0[:, 0], times, 9.81, 1.0)
    assert (traj[:, :, 0] - small).abs().max() < 0.01 * y0[0, 0]  # sin θ ≈ θ: simple harmonic motion
