"""Physics checks, written first (TDD): what any correct Bloch-equation code must satisfy.

Run with `pytest problems/medical/bloch_spin_echo/test.py`. Each test is a physical law: free precession keeps |M| =
1 and turns at the off-resonance frequency, a 90° pulse tips M into the transverse plane, and the echo refocuses T2*
dephasing to an amplitude exp(−TE/T2).
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _bloch(M_init, dw, T1, T2, pulses, t_max, dt=0.02):
    eps = tensors.levi_civita(CPU)
    R, M0 = tensors.relaxation(T1, T2, CPU)
    return solve.solve(M_init, dw, eps, R, M0, pulses, t_max, dt)


def test_single_isochromat_free_precession():
    M_init = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    dw = torch.tensor([0.5], dtype=torch.float64)
    times, M = _bloch(M_init, dw, 1e12, 1e12, [], 20.0)
    norm = M[:, 0].norm(dim=-1)
    assert (norm - 1).abs().max() < 1e-6  # |M| = 1 without relaxation
    phase = torch.atan2(M[:, 0, 1], M[:, 0, 0])
    expected = torch.atan2(torch.sin(-0.5 * times), torch.cos(-0.5 * times))  # precesses at Δω (clockwise)
    assert torch.allclose(phase, expected, atol=1e-6)


def test_ninety_pulse_tips_into_transverse_plane():
    M_init = tensors.equilibrium_state(1, CPU)
    pulses = tensors.pulse_list(TE=20.0, n_echoes=0, duration=0.2)
    _, M = _bloch(M_init, torch.zeros(1, dtype=torch.float64), 1e12, 1e12, pulses, 0.2)
    final = M[-1, 0]
    assert abs(final[2].item()) < 1e-4  # RK4 rotation error only
    assert final[:2].norm().item() == pytest.approx(1.0, abs=1e-5)


def test_echo_at_TE_with_true_T2_amplitude():
    T2, TE, n = 80.0, 24.0, 300
    dw = tensors.off_resonance(n, 4.0, "lorentzian", CPU)
    pulses = tensors.pulse_list(TE, 1, 0.2)
    t90 = 0.1
    times, M = _bloch(tensors.equilibrium_state(n, CPU), dw, 800.0, T2, pulses, t90 + 1.3 * TE, dt=0.05)
    mag = equation.signal(M).abs()
    # FID decays much faster than T2: T2* < T2
    t2_star = solve.fid_decay_time(times, M, t90, TE)
    assert t2_star < 0.2 * T2
    i_half = int(((times - (t90 + TE / 4)).abs()).argmin())
    assert mag[i_half] < math.exp(-(TE / 4) / T2) * 0.5
    # echo peak at TE with amplitude ≈ exp(-TE/T2)
    t_echo, height = solve.echo_peaks(times, M, TE, 1, t90)
    assert t_echo[0].item() - t90 == pytest.approx(TE, abs=0.3)
    assert height[0].item() == pytest.approx(math.exp(-TE / T2), rel=0.03)
