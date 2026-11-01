"""Physics checks, written first (TDD): what any correct Doppler code must satisfy.

Run with `pytest problems/waves/doppler_effect/test.py`. The heard frequency is f c / (c ∓ v_s), and a supersonic source
draws a Mach cone with sin μ = 1/M.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.mark.parametrize("mach", [0.0, 0.3, 0.8])
def test_heard_frequencies_match_formula(mach):
    c, f, t_max = tensors.C_SOUND, 2.0, 10.0
    v_s = mach * c
    x_start = -0.5 * v_s * t_max
    t_e = tensors.emission_times(f, t_max, CPU)
    s = tensors.source_position(t_e, v_s, x_start)
    obs = tensors.observers(x_start, v_s, t_max, c, CPU)
    heard = equation.heard_frequency(solve.solve(t_e, s, obs, c, t_max)["arrivals"])
    ahead, behind = equation.doppler_formula(f, c, v_s)
    assert heard[0].item() == pytest.approx(f * c / (c - v_s), rel=1e-4)  # Doppler: wavefronts bunch up ahead
    assert heard[1].item() == pytest.approx(f * c / (c + v_s), rel=1e-4)  # and spread out behind
    assert (ahead, behind) == pytest.approx((heard[0].item(), heard[1].item()), rel=1e-4)


def test_mach_cone_angle():
    c, mach, t_max = tensors.C_SOUND, 1.6, 8.0
    v_s = mach * c
    t_e = tensors.emission_times(1.0, t_max, CPU).double()
    s = tensors.source_position(t_e, v_s, 0.0)
    radii = equation.front_radii(t_max, t_e, c)
    mu = equation.cone_half_angle(s, radii, tensors.source_position(torch.tensor(t_max, dtype=torch.float64), v_s, 0.0))
    assert torch.allclose(mu, torch.full_like(mu, math.asin(1 / mach)), atol=1e-9)  # Mach cone: sin μ = 1/M
    # subsonic: the source sits inside all of its fronts, so there is no cone
    s_sub = tensors.source_position(t_e, 0.5 * c, 0.0)
    now = tensors.source_position(torch.tensor(t_max, dtype=torch.float64), 0.5 * c, 0.0)
    assert len(equation.cone_half_angle(s_sub, radii, now)) == 0
