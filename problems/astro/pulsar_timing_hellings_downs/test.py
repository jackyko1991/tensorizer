"""Physics checks, written first (TDD): what any correct pulsar-timing-array code must satisfy.

Run with `pytest problems/astro/pulsar_timing_hellings_downs/test.py`. Each test is a physical law, not an
implementation detail: the Hellings–Downs curve has the right limits and minimum and equals the sky average of
antenna patterns, gravitational waves are transverse and traceless, and simulated correlations prefer HD.
"""

import math

import pytest
import torch

from . import equation, solve, tensors


def test_limits_and_minimum():
    c = torch.cos(torch.deg2rad(torch.tensor([0.0, 180.0], dtype=torch.float64)))
    assert torch.allclose(equation.hellings_downs(c), torch.tensor([0.5, 0.25], dtype=torch.float64))
    zeta = torch.linspace(0.0, 180.0, 180001, dtype=torch.float64)
    gamma = equation.hellings_downs(torch.cos(torch.deg2rad(zeta)))
    z_min, g_min = equation.hd_minimum()
    assert zeta[gamma.argmin()].item() == pytest.approx(z_min, abs=0.01)
    assert gamma.min().item() == pytest.approx(g_min, abs=1e-9)
    assert z_min == pytest.approx(82.48, abs=0.01)
    assert g_min == pytest.approx(0.5 - 1.5 * math.exp(-5 / 6))


def test_curve_equals_sky_integral_of_antenna_patterns():
    gen = torch.Generator().manual_seed(4)
    zeta = torch.tensor([5.0, 40.0, 82.5, 130.0, 175.0], dtype=torch.float64)
    numeric = solve.hd_from_sky_integral(zeta, 400_000, gen)
    exact = equation.hellings_downs(torch.cos(torch.deg2rad(zeta)))
    assert torch.allclose(numeric, exact, atol=0.006)  # HD is the sky-averaged antenna-pattern correlation


def test_polarisation_tensors_are_transverse_traceless_and_orthogonal():
    gen = torch.Generator().manual_seed(5)
    omega = tensors.random_directions(100, gen)
    e = equation.polarisation_tensors(omega, torch.rand(100, generator=gen, dtype=torch.float64) * math.pi)
    # GWs are traceless
    assert torch.allclose(torch.einsum("mAii->mA", e), torch.zeros(100, 2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(
        torch.einsum("mAij,mj->mAi", e, omega), torch.zeros(100, 2, 3, dtype=torch.float64), atol=1e-12
    )
    gram = torch.einsum("mAij,mBij->mAB", e, e)
    assert torch.allclose(gram, 2 * torch.eye(2, dtype=torch.float64).expand(100, 2, 2), atol=1e-12)


def test_antenna_patterns_sky_average_and_exact_power():
    gen = torch.Generator().manual_seed(6)
    pulsar = tensors.random_directions(1, gen)
    omega = tensors.random_directions(200_000, gen)
    psi = torch.rand(200_000, generator=gen, dtype=torch.float64) * math.pi
    f = equation.antenna_patterns(pulsar, omega, equation.polarisation_tensors(omega, psi))  # (M, 2, 1)
    power = f.pow(2).sum(1)[:, 0]
    cos_theta = omega @ pulsar[0]
    assert torch.allclose(power, 0.25 * (1 - cos_theta) ** 2, atol=1e-9)  # Σ_A F² = ¼(1 − Ω̂·p̂)², any ψ
    assert power.mean().item() == pytest.approx(1 / 3, abs=0.004)


def test_simulated_correlations_prefer_hellings_downs():
    gen = torch.Generator().manual_seed(7)
    n = 50
    pulsars = tensors.random_directions(n, gen)
    r = solve.simulate(pulsars, tensors.pulsar_distances(n, gen), tensors.time_grid(96, 15.0), 12, 500, 6, 0.3, gen)
    zeta, rho = solve.pair_correlations(r, equation.separation_cos(pulsars))
    edges = tensors.bin_edges(10)
    binned = solve.bin_by_angle(zeta, rho, edges)
    fits = solve.fit_models((edges[1:] + edges[:-1]) / 2, binned.mean(0), solve.bin_errors(binned))
    chi_hd, chi_mono, chi_dip = (fits[k][1] for k in ("Hellings–Downs", "monopole", "dipole"))
    assert chi_hd < chi_mono / 5 and chi_hd < chi_dip / 5  # quadrupolar GW background, not monopole or dipole
    assert 0.5 < fits["Hellings–Downs"][0] < 1.2  # amplitude ≈ 1, diluted by white noise
