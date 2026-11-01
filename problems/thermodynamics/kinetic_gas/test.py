"""Physics checks, written first (TDD): what any correct hard-disc gas code must satisfy.

Run with `pytest problems/thermodynamics/kinetic_gas/test.py`. Elastic collisions conserve kinetic energy, the wall
pressure obeys the ideal gas law pA = NkT, and speeds relax to the 2D Maxwell–Boltzmann distribution.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def run_gas(n=200, radius=0.005, t_max=2.0):
    x0 = tensors.initial_positions(n, radius, CPU)
    v0 = tensors.initial_velocities(n, 1.0, CPU)
    return solve.solve(x0, v0, tensors.BOX, radius, t_max), v0


def test_conserves_kinetic_energy_and_follows_ideal_gas_law(run_gas):
    (times, xs, vs, pressure), v0 = run_gas
    energy = equation.kinetic_energy(vs)
    assert torch.allclose(energy, energy[0].expand_as(energy), rtol=1e-4)  # elastic collisions conserve energy

    kT = equation.temperature(v0).item()
    area = (tensors.BOX - 2 * 0.005) ** 2
    assert math.isclose(pressure, 200 * kT / area, rel_tol=0.15)  # ideal gas law: pA = NkT


def test_speeds_relax_away_from_a_single_speed(run_gas):
    (_, _, vs, _), v0 = run_gas
    speed = vs[-1].norm(dim=1)
    kT = equation.temperature(v0).item()
    assert speed.std() > 0.3  # started at exactly 0
    assert math.isclose(speed.mean().item(), math.sqrt(math.pi * kT / 2), rel_tol=0.1)  # 2D Maxwell–Boltzmann mean
