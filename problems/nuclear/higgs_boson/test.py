"""Physics and statistics checks, written first (TDD): what any correct H → γγ search must satisfy.

Run with `pytest problems/nuclear/higgs_boson/test.py`. Each test is a law, not an implementation detail: the fit
recovers a signal injected into synthetic data, the Asimov q₀ equals the closed form so Wilks gives Z = √q₀,
background-only toys follow ½δ(0) + ½χ²₁, the Mexican-hat potential has its minimum at v/√2 with m_H² = 2λv², and
the fit to real ATLAS Open Data finds the published Higgs mass.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def window():
    data = tensors.load_histogram(tensors.DATA, 105.0, 160.0)
    return data, tensors.design(data["centers"], 105.0, 160.0, degree=3)


def test_fit_recovers_an_injected_signal():
    data, x = window()
    theta = torch.tensor([7.2, -1.2, 0.2, 0.0], dtype=torch.float64)  # a smooth falling background
    b = equation.background(theta, x) * data["widths"]  # (N,) events per bin
    shape = equation.signal_shape(data["edges"], torch.tensor(130.0, dtype=torch.float64), 1.8)  # (N,)
    asimov = b + 300.0 * shape  # expected counts, no noise
    fit = solve.fit_free_mass(asimov, x, data["edges"], data["widths"], sigma=1.8, m_start=127.0, n_sm=100.0)
    assert fit["m_h"] == pytest.approx(130.0, abs=0.02)  # Asimov data: exact recovery
    assert fit["mu"] == pytest.approx(3.0, rel=0.01)  # 300 events / 100 expected
    gen = torch.Generator().manual_seed(1)
    noisy = torch.poisson(asimov, generator=gen)
    fit = solve.fit_free_mass(noisy, x, data["edges"], data["widths"], sigma=1.8, m_start=127.0, n_sm=100.0)
    assert abs(fit["m_h"] - 130.0) < 3 * fit["m_err"]  # within the statistical error
    assert abs(fit["mu"] - 3.0) < 3 * fit["mu_err"]


def test_asimov_q0_matches_closed_form_and_wilks():
    data, x = window()
    theta = torch.tensor([7.0, -1.0, 0.0, 0.0], dtype=torch.float64)
    b = equation.background(theta, x) * data["widths"]
    s = 400.0 * equation.signal_shape(data["edges"], torch.tensor(125.0, dtype=torch.float64), 1.7)
    q0 = solve.q0_known_background(b + s, b, s)  # profile over μ only
    closed = 2 * ((b + s) * torch.log1p(s / b) - s).sum()  # Cowan–Cranmer–Gross–Vitells Asimov formula
    assert q0.item() == pytest.approx(closed.item(), rel=1e-6)
    z = equation.significance(q0)
    p = equation.p_value(q0)
    assert z.item() == pytest.approx(math.sqrt(q0.item()), rel=1e-9)  # Wilks: Z = √q₀
    assert p.item() == pytest.approx(0.5 * math.erfc(math.sqrt(q0.item() / 2)), rel=1e-9)


def test_background_only_toys_follow_half_chi2():
    data, x = window()
    theta = torch.tensor([7.2, -1.2, 0.2, 0.0], dtype=torch.float64)
    b = equation.background(theta, x) * data["widths"]
    toys = tensors.toys(b, 4000, seed=3)  # (K, N) background-only pseudo-experiments
    mass = torch.tensor([125.0], dtype=torch.float64)
    q0 = solve.q0_scan(toys, x, data["edges"], data["widths"], mass, sigma=1.7)[:, 0]  # (K,)
    assert (q0 == 0).double().mean().item() == pytest.approx(0.5, abs=0.03)  # half the toys fluctuate down
    assert (q0 > 1).double().mean().item() == pytest.approx(0.5 * math.erfc(math.sqrt(0.5)), abs=0.02)
    assert (q0 > 4).double().mean().item() == pytest.approx(0.5 * math.erfc(math.sqrt(2)), abs=0.008)


def test_mexican_hat_minimum_and_higgs_mass():
    mu2, lam = tensors.potential_parameters(125.2, tensors.V_EW)
    r = torch.linspace(0.0, 250.0, 25001, dtype=torch.float64)  # |φ| in GeV
    v_of_r = equation.mexican_hat(r.to(torch.complex128), mu2, lam)
    r_min = r[v_of_r.argmin()].item()
    assert r_min == pytest.approx(tensors.V_EW / math.sqrt(2), abs=0.02)  # ⟨|φ|⟩ = v/√2
    h = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)  # φ = (v + h)/√2
    pot = equation.mexican_hat((tensors.V_EW + h) / math.sqrt(2) + 0j, mu2, lam)
    (d1,) = torch.autograd.grad(pot, h, create_graph=True)
    (d2,) = torch.autograd.grad(d1, h)
    assert abs(d1.item()) < 1e-6  # stationary at the vacuum
    assert math.sqrt(d2.item()) == pytest.approx(125.2, rel=1e-9)  # curvature: m_H² = 2λv²
    assert 2 * lam * tensors.V_EW**2 == pytest.approx(125.2**2, rel=1e-12)
    phase = torch.exp(1j * torch.linspace(0, 2 * math.pi, 9, dtype=torch.float64))  # the vacuum circle
    ring = equation.mexican_hat(tensors.V_EW / math.sqrt(2) * phase, mu2, lam)
    assert torch.allclose(ring, ring[0].expand_as(ring))  # every phase is an equally good vacuum


def test_real_data_finds_the_published_higgs_mass():
    data, x = window()
    n_sm = data["sm_signal"].sum().item()
    m_sim, sigma = solve.fit_resolution(data["edges"], data["sm_signal"])  # peak width from the ATLAS simulation
    assert m_sim == pytest.approx(125.0, abs=0.1)  # the simulation was generated at m_H = 125 GeV
    fit = solve.fit_free_mass(data["counts"], x, data["edges"], data["widths"], sigma, m_start=127.0, n_sm=n_sm)
    m_pdg, err_pdg = tensors.M_H_PDG
    assert abs(fit["m_h"] - m_pdg) < 2 * math.hypot(fit["m_err"], err_pdg)  # within the fit error (2σ)
    assert fit["mu"] > 2 * fit["mu_err"]  # an excess (about 2.5σ local in this small 10 fb⁻¹ sample)
    assert abs(fit["mu"] - 1.0) < 2 * fit["mu_err"]  # compatible with the Standard Model rate μ = 1
