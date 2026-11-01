"""Physics checks, written first (TDD): what any correct Kepler-orbit code must satisfy.

Run with `pytest problems/astro/kepler_orbits/test.py`. Each test is a physical law, not an implementation detail:
Kepler's third law holds, areal velocity, energy and angular momentum are conserved, and the energy fixes the semi-
major axis.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def orbits():
    GM = tensors.GM_SUN
    a, e = tensors.orbit_shapes(4, 0.6)
    y0 = tensors.initial_state(a.double(), e.double(), GM, CPU)
    times, traj = solve.solve(y0, GM, t_max=3.5, dt=0.0005)
    return GM, a.double(), times, traj


def test_third_law(orbits):
    GM, a, times, traj = orbits
    T = solve.measured_periods(times, traj)
    assert torch.isfinite(T).all()
    # Kepler's third law: T²/a³ = 4π²/GM
    assert torch.allclose(T**2 / a**3, torch.full_like(a, 4 * math.pi**2 / GM), rtol=1e-3)


def test_areal_velocity_energy_and_angular_momentum_are_conserved(orbits):
    GM, _, _, traj = orbits
    for q in (equation.areal_velocity(traj), equation.energy(traj, GM), equation.angular_momentum(traj)):
        # Kepler's second law, energy and angular momentum conserved
        assert torch.allclose(q, q[0].expand_as(q), rtol=1e-5)


def test_semi_major_axis_from_energy(orbits):
    GM, a, _, traj = orbits
    assert torch.allclose(equation.semi_major_axis(traj[0], GM), a, rtol=1e-9)  # vis-viva: E = −GM/2a
