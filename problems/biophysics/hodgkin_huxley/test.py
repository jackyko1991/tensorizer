"""Physics checks, written first (TDD): what any correct Hodgkin–Huxley code must satisfy.

Run with `pytest problems/biophysics/hodgkin_huxley/test.py`. Each test is a physical law: the resting potential is
a fixed point, gates stay in [0, 1], spikes stay below E_Na, firing starts at a finite rate (type II), and warmer
axons fire faster.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def _hh_run(currents, temp=6.3, t_end=150.0):
    g, E = tensors.channels(CPU)
    I_ext = torch.tensor(currents)
    y0 = tensors.initial_state(len(I_ext), CPU)
    times, traj = solve.solve(y0, I_ext, g, E, tensors.C_M, tensors.temperature_factor(temp), t_end)
    spikes = solve.spike_times(times, traj[:, 0])
    return traj, spikes, solve.firing_rate(spikes, t_end)


def test_resting_state_is_a_fixed_point():
    g, E = (v.to(F64) for v in tensors.channels(CPU))
    V = torch.tensor([-65.0], dtype=F64)
    y0 = torch.cat([V[None], equation.steady_state(V)])
    m, h, n = y0[1:, 0].tolist()
    assert abs(m - 0.0529) < 1e-3 and abs(h - 0.5961) < 1e-3 and abs(n - 0.3177) < 1e-3
    dy = equation.rhs(y0, torch.zeros(1, dtype=F64), g, E, tensors.C_M, 1.0)
    # E_L = −54.387 mV (the classic rounded value) makes −65 mV the rest to within ~0.01 mV
    assert dy.abs().max() < 1e-2
    traj, spikes, _ = _hh_run([0.0], t_end=50.0)
    assert (traj[:, 0] + 65.0).abs().max() < 0.05 and len(spikes[0]) == 0


def test_rate_functions_are_continuous_at_removable_singularities():
    for V0 in (-40.0, -55.0):
        V = torch.tensor([V0 - 1e-3, V0, V0 + 1e-3], dtype=F64)
        alpha, _ = equation.rates(V)
        assert torch.isfinite(alpha).all()
        assert (alpha[:, 1] - 0.5 * (alpha[:, 0] + alpha[:, 2])).abs().max() < 1e-6
    assert math.isclose(equation.rates(torch.tensor([-40.0], dtype=F64))[0][0].item(), 1.0)  # α_m(−40) = 1/ms


def test_threshold_and_f_i_curve():
    currents = [0.0, 2.0, 4.0, 10.0, 15.0, 20.0]
    traj, spikes, rate = _hh_run(currents)
    gates = traj[:, 1:]
    assert gates.min() >= 0.0 and gates.max() <= 1.0
    n = [len(s) for s in spikes]
    assert n[0] == 0 and n[1] == 0  # below threshold: no spike
    assert n[2] == 1 and rate[2] == 0  # a single spike, then a new resting state
    assert 35.0 < traj[:, 0].max() < 50.0  # spike peak below E_Na = 50 mV
    assert 60.0 < rate[3] < 75.0  # classic ≈ 68 Hz at 10 µA/cm²
    assert (rate[3:][1:] > rate[3:][:-1]).all()  # rate rises with current


@pytest.mark.slow  # long training/sampling run: CI only, skipped by the in-app TDD check
def test_type_ii_onset_and_temperature():
    _, _, rate = _hh_run([6.0, 6.5, 7.0])
    first = rate[rate > 0]
    assert len(first) > 0 and first[0] > 40.0  # repetitive firing starts at a finite rate (Hopf, type II)
    _, _, warm = _hh_run([10.0], temp=15.0)
    _, _, cold = _hh_run([10.0], temp=6.3)
    assert warm[0] > 1.3 * cold[0]  # faster gates ⇒ faster firing
