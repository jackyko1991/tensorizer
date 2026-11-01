"""Physics checks, written first (TDD): what any correct dispersive-wave code must satisfy.

Run with `pytest problems/waves/dispersive_waves/test.py`. On deep water the group velocity is half the phase velocity,
and the packet conserves energy while it spreads.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_deep_water_group_velocity_is_half_phase_velocity():
    g, k0 = tensors.G, 2 * math.pi / 12.0
    x, k, _ = tensors.grid(2048, CPU)
    a0 = tensors.wave_packet(x, tensors.X0, 30.0, k0)  # wide packet: narrow spectrum
    times, a = solve.solve(a0, k, g, t_max=40.0, n_times=401)
    v_g = solve.envelope_speed(times, solve.centroid(x, a))
    v_p = solve.crest_speed(x, a, times)
    assert v_p == pytest.approx(equation.phase_velocity(k0, g), rel=1e-2)
    assert v_g == pytest.approx(equation.group_velocity(k0, g), rel=1e-2)
    assert v_g / v_p == pytest.approx(0.5, rel=2e-2)  # deep water ω = √(gk): v_g = v_p / 2


def test_dispersive_packet_conserves_energy_and_spreads():
    x, k, _ = tensors.grid(2048, CPU)
    a0 = tensors.wave_packet(x, tensors.X0, 8.0, 2 * math.pi / 10.0)
    _, a = solve.solve(a0, k, tensors.G, t_max=60.0, n_times=7)
    E = equation.wave_energy(a)
    assert torch.allclose(E, E[0].expand_as(E), rtol=1e-10)
    assert a[-1].abs().max() < 0.8 * a[0].abs().max()  # dispersion lowers and widens the envelope
