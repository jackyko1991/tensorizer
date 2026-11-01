"""Physics checks, written first (TDD): what any correct beats code must satisfy.

Run with `pytest problems/waves/beats_superposition/test.py`. The beat frequency is |f₁ − f₂|, the envelope bounds the
signal, and v_g / v_p equals the dispersion power law exponent.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_beat_frequency_is_frequency_difference():
    omega = tensors.angular_frequencies(1.0, 1.25, 2, CPU)
    k = tensors.wavenumbers(omega, "none")
    A = tensors.amplitudes(2, CPU)
    x, t = tensors.grids(10.0, 40.0, CPU)
    res = solve.solve(A, k, omega, x, t)
    dt = (t[1] - t[0]).item()
    assert solve.beat_frequency(res["trace"], dt, f_min=1.0) == pytest.approx(0.25, abs=0.01)
    # envelope period 1/|f1 - f2| = 4 s: the envelope at x = 0 is |cos(π Δf t)|, back to 1 after 4 s
    env = res["trace_env"]
    expected = torch.abs(torch.cos(math.pi * 0.25 * t))
    assert torch.allclose(env, expected, atol=1e-4)
    assert torch.all(res["trace"].abs() <= env + 1e-5)  # the envelope bounds the superposition


@pytest.mark.parametrize("dispersion, ratio", [("none", 1.0), ("water", 0.5), ("capillary", 1.5)])
def test_group_over_phase_velocity_equals_dispersion_power(dispersion, ratio):
    omega = tensors.angular_frequencies(1.0, 1.01, 2, CPU).double()
    k = tensors.wavenumbers(omega, dispersion)
    assert equation.group_velocity(k, omega) / equation.phase_velocity(k, omega) == pytest.approx(
        ratio, rel=1e-2
    )  # ω ∝ k^p gives v_g / v_p = p
