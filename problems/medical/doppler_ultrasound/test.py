"""Physics checks, written first (TDD): what any correct Doppler ultrasound code must satisfy.

Run with `pytest problems/medical/doppler_ultrasound/test.py`. Each test is a physical law: the shift is 2 f0 v cos
θ / c and vanishes for a perpendicular beam, a parabolic profile has mean frequency f_max / 2, and frequencies above
PRF / 2 alias.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F0, PRF, R = 5e6, 10e3, 3e-3


def _steady_spectrum(v_max: float, theta: float, n: int, prf: float = PRF, duration: float = 1.0):
    """Average spectrum of n scatterers in steady flow with centreline velocity v_max (parabolic if n > 1)."""
    t = torch.arange(int(duration * prf)) / prf
    scat = tensors.scatterers(n, R, 20.0, CPU, seed=3)
    if n == 1:
        scat["r"] = torch.zeros(1)  # on the axis: moves at v_max
    _, s = solve.simulate(torch.full_like(t, v_max), scat, R, F0, theta, prf)
    _, freqs, power = solve.spectrogram(s, prf, n_fft=256, hop=64)
    return freqs, power.mean(1)


def test_single_scatterer_shift_matches_formula():
    theta = math.radians(45)
    for v in (0.3, 0.8, -0.5):
        freqs, spec = _steady_spectrum(v, theta, 1)
        expected = 2 * F0 * v * math.cos(theta) / equation.C_TISSUE
        assert abs(equation.doppler_shift(torch.tensor(v), F0, theta).item() - expected) < 1e-6 * abs(expected) + 1e-3
        assert abs(freqs[spec.argmax()].item() - expected) <= (freqs[1] - freqs[0]).item()
        # the inverse relation recovers the speed
        assert abs(equation.velocity_from_frequency(torch.tensor(expected), F0, theta).item() - v) < 1e-5


def test_perpendicular_beam_gives_zero_shift():
    assert abs(equation.doppler_shift(torch.tensor(1.0), F0, math.pi / 2).item()) < 1e-9
    freqs, spec = _steady_spectrum(1.0, math.pi / 2, 1)
    assert abs(freqs[spec.argmax()].item()) < 1e-9


def test_max_spectral_frequency_is_centreline_velocity():
    theta, v_max = math.radians(60), 1.0
    freqs, spec = _steady_spectrum(v_max, theta, 2000)
    f_max = equation.doppler_shift(torch.tensor(v_max), F0, theta).item()  # 3247 Hz
    plateau = spec[(freqs > 0.2 * f_max) & (freqs < 0.8 * f_max)].mean()
    edge = freqs[spec > 0.1 * plateau].max().item()  # highest frequency with real signal
    assert abs(edge - f_max) <= 2 * (freqs[1] - freqs[0]).item()


def test_parabolic_profile_mean_frequency_is_half_max():
    theta, v_max = math.radians(60), 1.0
    freqs, spec = _steady_spectrum(v_max, theta, 2000)
    f_max = equation.doppler_shift(torch.tensor(v_max), F0, theta).item()
    f_mean = solve.mean_frequency(freqs, spec[:, None]).item()
    assert abs(f_mean / f_max - 0.5) < 0.04
    # the same from the velocities themselves: uniform density over the disc ⇒ <1 - r²/R²> = 1/2
    v = equation.velocity_field(torch.ones(1), tensors.scatterers(20000, R, 20.0, CPU)["r"], R)
    assert abs(v.mean().item() - 0.5) < 0.01


def test_aliasing_wraps_above_nyquist():
    theta, prf = 0.0, 4e3
    v = 0.8  # f_D = 2 · 5 MHz · 0.8 / 1540 ≈ 5.19 kHz > PRF/2 = 2 kHz
    f_true = equation.doppler_shift(torch.tensor(v), F0, theta).item()
    f_alias = equation.alias(torch.tensor(f_true), prf).item()
    assert -prf / 2 <= f_alias < prf / 2
    assert abs((f_true - f_alias) / prf - round((f_true - f_alias) / prf)) < 1e-6  # differs by a multiple of PRF
    freqs, spec = _steady_spectrum(v, theta, 1, prf=prf)
    assert abs(freqs[spec.argmax()].item() - f_alias) <= (freqs[1] - freqs[0]).item()
    # below the Nyquist limit nothing changes
    assert equation.alias(torch.tensor(1500.0), prf).item() == 1500.0


def test_cardiac_waveform_has_systolic_peak():
    t = tensors.sample_times(1000.0, 60.0, 1, CPU)
    v = tensors.cardiac_waveform(t, 1.0, 60.0)
    assert abs(v.max().item() - 1.0) < 0.05 and 0.08 < t[v.argmax()].item() < 0.16
    assert v.min().item() < 0  # brief reverse flow
