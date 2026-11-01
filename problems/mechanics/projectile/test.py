"""Physics checks, written first (TDD): what any correct projectile code must satisfy.

Run with `pytest problems/mechanics/projectile/test.py`. Without drag the range is v² sin 2θ / g and energy is
conserved; drag only removes energy, so it shortens the range and lowers the best launch angle below 45°.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_range_without_drag_matches_textbook():
    g = tensors.gravity(CPU)
    angles = tensors.launch_angles(5)
    y0 = tensors.initial_state(20.0, angles, CPU)
    _, traj = solve.solve(y0, g, k=0.0, t_max=5.0, dt=0.001)
    expected = 20.0**2 * torch.sin(2 * angles) / 9.81
    assert torch.allclose(solve.ranges(traj), expected, rtol=5e-3)  # R = v² sin 2θ / g


def test_energy_is_conserved_without_drag():
    g = tensors.gravity(CPU)
    y0 = tensors.initial_state(15.0, torch.tensor([math.radians(60.0)]), CPU)
    _, traj = solve.solve(y0, g, k=0.0, t_max=1.0)  # still airborne
    e = equation.energy(traj, g)
    assert torch.allclose(e, e[0].expand_as(e), rtol=1e-4)  # float32 round-off


def test_drag_shortens_range_and_lowers_best_angle():
    g = tensors.gravity(CPU)
    angles = tensors.launch_angles(13)
    y0 = tensors.initial_state(30.0, angles, CPU)
    _, traj = solve.solve(y0, g, k=0.05, t_max=8.0)
    r = solve.ranges(traj)
    assert r.max() < 30.0**2 / 9.81  # drag removes energy: shorter than the vacuum range
    assert angles[r.argmax()] < math.radians(45.0)
