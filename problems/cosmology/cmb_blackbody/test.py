"""Checks written first (TDD): what any correct fit of the CMB spectrum must satisfy.

Run with `pytest problems/cosmology/cmb_blackbody/test.py`. The Planck law must integrate to σT⁴ (Stefan–Boltzmann),
peak where Wien says and fall to Rayleigh–Jeans at low frequency; the Compton y-distortion conserves photon number
and changes sign at x = 3.83; the fit recovers known parameters from synthetic data; and on the real FIRAS spectrum it
gives T ≈ 2.725 K with χ² per degree of freedom near 1 and distortions consistent with zero.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

F64 = torch.float64
SIGMA_SB = 5.670374419e-8  # W m⁻² K⁻⁴ (exact in SI since 2019)


def test_planck_law_integrates_to_stefan_boltzmann():
    T = torch.tensor([2.725, 10.0, 300.0], dtype=F64)
    nu = torch.linspace(1e-4, 1.0, 200_001, dtype=F64)[:, None] * 40 * T / equation.C2  # (F, 3) up to x = 40
    intensity = equation.planck(nu, T)  # (F, 3) I_ν in MJy/sr, on a wavenumber grid ν̃ in cm⁻¹
    # flux through a surface = π ∫ I_ν dν, with dν = c dν̃; 1 MJy = 1e-20 W m⁻² Hz⁻¹ and ν̃ in cm⁻¹ → m⁻¹
    flux = math.pi * torch.trapezoid(intensity * 1e-20, nu * 100 * equation.C_LIGHT, dim=0)
    assert torch.allclose(flux, SIGMA_SB * T**4, rtol=1e-6)


def test_wien_peak_and_rayleigh_jeans_limit():
    T = torch.tensor(2.725, dtype=F64)
    nu = torch.linspace(1.0, 15.0, 140_001, dtype=F64)
    peak = nu[equation.planck(nu, T).argmax()]
    # I_ν peaks at x = hν/kT = 2.821439 (root of 3(1 − e^−x) = x): 5.35 cm⁻¹ = 160 GHz for 2.725 K
    assert peak.item() == pytest.approx(2.821439 * T.item() / equation.C2, rel=1e-5)
    low = torch.tensor([1e-4, 1e-3], dtype=F64)
    rj = 2 * equation.K_B * T * (low * 100) ** 2 * 1e20  # 2ν²kT/c² = 2kT ν̃² with ν = c ν̃ (ν̃ in m⁻¹), in MJy/sr
    assert torch.allclose(equation.planck(low, T), rj, rtol=1e-3)  # x ≪ 1: classical equipartition


def test_y_distortion_conserves_photons_and_has_its_null_at_x_383():
    x = torch.linspace(1e-4, 40, 400_001, dtype=F64)
    shape = equation.y_shape(x)  # Δn / y, the Kompaneets solution for a small Compton y
    assert torch.trapezoid(x**2 * shape, x).abs() < 1e-6  # ∫ x² Δn dx = 0: scattering moves photons, never makes them
    crossing = x[(shape[:-1] * shape[1:] < 0).nonzero()[0, 0]]
    assert crossing.item() == pytest.approx(3.830, abs=2e-3)  # the 217 GHz null of the Sunyaev–Zel'dovich effect


def test_number_and_energy_density_of_the_cmb():
    T = torch.tensor(2.7255, dtype=F64)
    assert equation.photon_number_density(T).item() == pytest.approx(410.7, abs=0.2)  # cm⁻³
    assert equation.energy_density(T).item() == pytest.approx(0.2606, abs=5e-4)  # eV cm⁻³
    # the same numbers by integrating the spectrum: n = (4π/c) ∫ I_ν / (hν) dν
    nu = torch.linspace(1e-4, 40 * T.item() / equation.C2, 200_001, dtype=F64)
    i_si = equation.planck(nu, T) * 1e-20  # W m⁻² Hz⁻¹ sr⁻¹
    freq = nu * 100 * equation.C_LIGHT  # Hz
    n_quad = 4 * math.pi / equation.C_LIGHT * torch.trapezoid(i_si / (equation.H_PLANCK * freq), freq) * 1e-6
    assert n_quad.item() == pytest.approx(equation.photon_number_density(T).item(), rel=1e-5)


def test_fit_recovers_known_parameters_from_synthetic_data():
    data = tensors.load_firas(torch.device("cpu"))
    truth = torch.tensor([[2.7, 0.0, 0.0], [2.7, 3e-4, 0.0], [2.7, 0.0, 5e-5], [2.7, 2e-4, 3e-5]], dtype=F64)
    fake = equation.spectrum(data["nu"], truth)  # (4, F) noiseless
    g = torch.Generator().manual_seed(1)
    noisy = fake + data["sigma"] * torch.randn(fake.shape, generator=g, dtype=F64)
    fit = solve.fit(data["nu"], noisy, data["sigma"], T_init=2.4, steps=40)
    err = fit["params"] - truth
    assert (err.abs() < 4 * fit["sigma"] + 1e-12).all()  # within 4σ for every model and parameter
    assert torch.allclose(fit["params"][:, 0], truth[:, 0], atol=1e-4)  # T to better than 0.1 mK


def test_firas_is_a_blackbody_at_2725_k():
    data = tensors.load_firas(torch.device("cpu"))
    fit = solve.fit(data["nu"], data["intensity"], data["sigma"], T_init=2.5, steps=40)
    T, mu, y = fit["params"].T  # models: T only, T + μ, T + y, T + μ + y
    assert torch.allclose(T, torch.tensor(2.725, dtype=F64), atol=2e-4)
    dof = fit["dof"]
    assert (fit["chi2"] / dof > 0.5).all() and (fit["chi2"] / dof < 1.5).all()  # FIRAS: χ² ≈ 45 for 43 points
    assert mu[0] == 0 and y[0] == 0 and y[1] == 0 and mu[2] == 0  # parameters not in a model stay at zero
    assert abs(mu[1]) < 2 * fit["sigma"][1, 1] and abs(y[2]) < 2 * fit["sigma"][2, 2]  # consistent with no distortion
    mu_lim, y_lim = solve.upper_limits(fit)
    assert 3e-5 < mu_lim < 2e-4  # Fixsen et al. 1996: |μ| < 9e-5 (95 %)
    assert 2e-6 < y_lim < 3e-5  # Fixsen et al. 1996: |y| < 1.5e-5 (95 %; theirs adds correlated errors and systematics)
