"""Physics checks, written first (TDD): what any correct N-body code must satisfy.

Run with `pytest problems/mechanics/n_body_gravity/test.py`. Each test is a physical law, not an implementation
detail: energy and momentum are conserved, two bodies obey Kepler's third law, and the figure-8 is periodic.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.mark.parametrize("scenario", ["3D cluster", "figure-8", "star and planets", "binary and planet"])
def test_energy_and_momentum_are_conserved(scenario):
    m, x0, v0 = tensors.initial_state(scenario, CPU)
    _, xs, vs = solve.solve(m, x0, v0, tensors.G, eps=0.0, t_max=6.0, dt=0.002)
    e0 = equation.energy(xs[0], vs[0], m, tensors.G, 0.0)
    e1 = equation.energy(xs[-1], vs[-1], m, tensors.G, 0.0)
    assert abs((e1 - e0) / e0) < 1e-5  # no external force: total energy is constant
    p = equation.momentum(vs, m)  # (T, 3)
    assert p.abs().max() < 1e-10  # internal forces cancel pairwise (Newton's third law): total momentum stays 0


def test_two_body_period_matches_kepler():
    m1, m2, a = 1.0, 0.3, 1.5
    m, x0, v0 = tensors.two_body_circular(m1, m2, a, CPU)
    period = 2 * math.pi * math.sqrt(a**3 / (tensors.G * (m1 + m2)))  # Kepler's third law: T² = 4π² a³ / G(m1+m2)
    times, xs, _ = solve.solve(m, x0, v0, tensors.G, eps=0.0, t_max=1.5 * period, dt=0.001)
    rel = xs[:, 1] - xs[:, 0]  # (T, 3) separation vector
    angle = torch.atan2(rel[:, 1], rel[:, 0])
    step = (angle.diff() + math.pi) % (2 * math.pi) - math.pi  # angle swept per step, no 2π jumps
    swept = torch.cat([torch.zeros(1, dtype=step.dtype), torch.cumsum(step, 0)])  # (T,) increasing
    measured = times[torch.searchsorted(swept, torch.tensor([2 * math.pi], dtype=swept.dtype))].item()
    assert measured == pytest.approx(period, rel=1e-3)
    assert torch.allclose(rel.norm(dim=-1), torch.tensor(a, dtype=rel.dtype), rtol=1e-6)  # stays circular


def test_figure_eight_is_periodic():
    m, x0, v0 = tensors.figure_eight(CPU)
    _, xs, _ = solve.solve(m, x0, v0, tensors.G, eps=0.0, t_max=6.32591398, dt=0.001)
    assert torch.allclose(xs[-1], x0, atol=1e-3)  # back to the start after one period


def test_cluster_is_three_dimensional_and_virial():
    m, x, v = tensors.cluster(device=CPU)
    assert x[:, 2].std() > 0.3  # not a planar configuration
    kinetic = 0.5 * torch.einsum("a,ai,ai->", m, v, v)
    potential = equation.energy(x, v, m, tensors.G, 0.0) - kinetic
    assert (2 * kinetic / -potential).item() == pytest.approx(1.0, rel=1e-9)  # virial equilibrium 2K = −U
