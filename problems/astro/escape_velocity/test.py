"""Physics checks, written first (TDD): what any correct escape-velocity code must satisfy.

Run with `pytest problems/astro/escape_velocity/test.py`. Each test is a physical law, not an implementation detail:
v_esc = √(2GM/R) gives exactly zero energy, a body escapes iff its energy is positive, and the highest point follows
from energy conservation.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_escape_speed_formula_and_zero_energy():
    v_esc = equation.escape_speed(tensors.GM, tensors.R)
    assert v_esc == pytest.approx(math.sqrt(2))
    y0 = tensors.initial_state(torch.tensor([1.0], dtype=torch.float64), v_esc, 0.0, CPU)
    assert equation.energy(y0, tensors.GM).abs().item() < 1e-12  # at v_esc the total energy is exactly zero
    assert equation.dark_star_radius(1.327e20, 2.998e8) == pytest.approx(2953, rel=1e-3)  # the Sun: about 3 km


def test_energy_sign_decides_escape():
    f = torch.tensor([0.6, 0.8, 0.9, 1.1, 1.3], dtype=torch.float64)
    v_esc = equation.escape_speed(tensors.GM, tensors.R)
    y0 = tensors.initial_state(f, v_esc, 0.0, CPU)
    _, traj = solve.solve(y0, tensors.GM, tensors.R, t_max=40.0, dt=0.005)
    E = equation.energy(y0, tensors.GM)
    assert torch.equal(solve.escaped(traj, tensors.R), E > 0)  # unbound iff E > 0


def test_max_height_matches_energy_conservation():
    f = torch.tensor([0.5, 0.7, 0.85], dtype=torch.float64)
    v_esc = equation.escape_speed(tensors.GM, tensors.R)
    y0 = tensors.initial_state(f, v_esc, 0.0, CPU)
    _, traj = solve.solve(y0, tensors.GM, tensors.R, t_max=20.0, dt=0.002)
    expected = equation.max_radius(f * v_esc, tensors.GM, tensors.R)
    assert torch.allclose(solve.highest_point(traj), expected, rtol=1e-4)  # energy conservation fixes the turning point
