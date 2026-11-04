"""Physics checks, written first (TDD): what any correct model of the photoelectric effect must satisfy.

Run with `pytest problems/quantum/photoelectric_effect/test.py`. Einstein's light-quantum law eV_s = hν − φ:
no electron leaves below the threshold ν0 = φ/h, the stopping potential grows with slope h/e and does not depend on
the intensity (only the current does), and Millikan's 1916 sodium measurements give back Planck's constant.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

PHI = 2.28  # eV, clean sodium


def test_no_emission_below_threshold():
    nu0 = equation.threshold_frequency(PHI)  # 10^14 Hz
    nu = torch.tensor([0.5 * nu0, 0.99 * nu0, 1.01 * nu0, 2 * nu0], dtype=torch.float64)
    depth = tensors.electron_depths(len(nu), 20000, width=3.0, seed=0)
    K = equation.kinetic_energy(nu, PHI, depth)  # (F, N)
    emitted = (K > 0).sum(dim=1)
    assert emitted[0] == 0 and emitted[1] == 0  # however bright: below ν0 a photon cannot pay the work function
    assert emitted[2] > 0 and emitted[3] > emitted[2]
    V = torch.linspace(-1, 3, 50, dtype=torch.float64)
    current = equation.photocurrent(V[None, :], nu[:, None], PHI, flux=1e6, width=3.0)
    assert current[:2].abs().max() == 0


def test_photon_energy_is_conserved():
    nu = tensors.frequencies(6.0, 14.0, 9)
    depth = tensors.electron_depths(len(nu), 5000, width=3.0, seed=1)
    K = equation.kinetic_energy(nu, PHI, depth)
    # hν = φ + ε + K for every electron: K never exceeds hν − φ
    assert (K <= equation.photon_energy(nu)[:, None] - PHI + 1e-12).all()


def test_slope_is_h_over_e_on_synthetic_data():
    nu = tensors.frequencies(6.0, 14.0, 12)
    depth = tensors.electron_depths(len(nu), 40000, width=3.0, seed=2)
    Vs = solve.measured_stopping_potential(equation.kinetic_energy(nu, PHI, depth))  # (F,)
    fit = solve.fit_line(nu, Vs)
    assert fit["slope"] == pytest.approx(equation.H_OVER_E, rel=2e-3)  # slope = h/e
    assert -fit["intercept"] == pytest.approx(PHI, abs=0.01)  # intercept = −φ/e


def test_stopping_potential_does_not_depend_on_intensity():
    nu = torch.tensor([8.0, 11.0], dtype=torch.float64)
    sim = solve.simulate(nu, PHI, intensities=torch.tensor([0.25, 1.0, 4.0]), n_photons=40000, width=3.0, seed=3)
    Vs = sim["V_stop"]  # (I, F): one stopping potential per intensity and frequency
    exact = equation.stopping_potential(nu, PHI)
    assert torch.allclose(Vs, exact.expand_as(Vs), atol=0.02)  # brighter light: same eV_s = hν − φ
    # ... while the saturation current scales with the intensity (number of photons)
    I_sat = sim["current"][:, :, 0]  # (I, F) at the most negative (accelerating) voltage
    ratio = I_sat / I_sat[1:2]
    assert torch.allclose(ratio, torch.tensor([[0.25], [1.0], [4.0]], dtype=ratio.dtype).expand_as(ratio), rtol=0.03)
    # the classical wave picture predicts the opposite: kinetic energy growing with intensity
    Kc = equation.classical_kinetic_energy(torch.tensor([0.25, 1.0, 4.0]), K1=1.0)
    assert Kc[2] / Kc[0] == pytest.approx(16.0)  # 16× the intensity, 16× the energy, at any ν


def test_classical_delay_is_long_for_dim_light():
    # energy collected over an atom-sized area: minutes for 10 mW/m²
    # — yet no delay is observed (< 3 ns, Lawrence & Beams 1928)
    t = equation.classical_delay(S=1e-2, phi=PHI, area=math.pi * (1e-10) ** 2)
    assert t > 60.0


def test_millikan_1916_gives_plancks_constant():
    nu, V = tensors.millikan_sodium()
    fit = solve.fit_line(nu, V)
    h = fit["slope"] * 1e-14 * equation.E_CHARGE  # V·s × C = J·s
    assert h == pytest.approx(equation.H_PLANCK, rel=0.02)  # Millikan: h within ~1 % of today's value
    nu0 = -fit["intercept"] / fit["slope"]
    assert 4.0 < nu0 < 4.8  # threshold near 4.4×10^14 Hz (red light, λ0 ≈ 680 nm)
