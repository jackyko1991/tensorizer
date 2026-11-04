"""Physics checks, written first (TDD): what any correct rotation-curve code must satisfy.

Run with `pytest problems/cosmology/dark_matter_rotation/test.py`. Each test is a law or a classic result: the
Freeman exponential disk peaks at 2.2 R_d, the NFW enclosed mass has a closed form, far from any mass V falls like
r^{-1/2} (Kepler), independent components add in quadrature, deep MOND gives V⁴ = G M a₀ (baryonic Tully–Fisher), and
the batched fit recovers the parameters of synthetic galaxies. On the real SPARC galaxies, halos beat baryons alone.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64
G = tensors.G_NEWTON


def test_exponential_disk_peaks_at_2_2_scale_lengths():
    Rd = 3.0
    R = torch.linspace(0.05, 20, 4000, dtype=F64)
    V = equation.exponential_disk_velocity(R, mass=5e10, Rd=Rd)
    assert float(R[V.argmax()] / Rd) == pytest.approx(2.15, abs=0.02)  # Freeman 1970: peak at 2.15 R_d
    far = torch.tensor([60 * Rd], dtype=F64)
    kepler = math.sqrt(G * 5e10 / float(far))  # far out the disk looks like a point mass
    assert float(equation.exponential_disk_velocity(far, 5e10, Rd)) == pytest.approx(kepler, rel=2e-3)


def test_nfw_enclosed_mass_matches_integral_of_density():
    rho_s, r_s = 1e7, 8.0  # M_sun/kpc³, kpc
    r = torch.linspace(1e-4, 60, 200001, dtype=F64)
    x = r / r_s
    rho = rho_s / (x * (1 + x) ** 2)  # NFW density
    M_num = torch.cumulative_trapezoid(4 * math.pi * r**2 * rho, r)
    M_exact = 4 * math.pi * rho_s * r_s**3 * (torch.log(1 + x) - x / (1 + x))
    assert torch.allclose(M_num[1000::5000], M_exact[1:][1000::5000], rtol=1e-4)
    V_s = math.sqrt(4 * math.pi * G * rho_s * r_s**2)
    far = r[1:] > 0.05  # (very close to the centre ln(1+x) − x/(1+x) suffers from cancellation)
    V = equation.nfw_velocity(r[1:][far], V_s, r_s)
    assert torch.allclose(V**2, G * M_exact[1:][far] / r[1:][far], rtol=1e-8)  # V² = G M(<r)/r


def test_point_mass_falls_off_like_kepler():
    r = torch.tensor([1.0, 4.0, 16.0], dtype=F64)
    V = equation.point_mass_velocity(r, 1e10)
    assert torch.allclose(V[1:] / V[:-1], torch.full((2,), 0.5, dtype=F64))  # V ∝ r^{-1/2}


def test_components_add_in_quadrature():
    g = torch.tensor([[10.0, -5.0]], dtype=F64)
    d = torch.tensor([[30.0, 40.0]], dtype=F64)
    b = torch.tensor([[0.0, 20.0]], dtype=F64)
    halo = torch.tensor([[50.0, 60.0]], dtype=F64)
    ups_d, ups_b = torch.tensor([0.5], dtype=F64), torch.tensor([0.7], dtype=F64)
    V2_bar = equation.baryonic_v2(g, d, b, ups_d, ups_b)
    assert torch.allclose(V2_bar, torch.tensor([[100 + 0.5 * 900, -25 + 0.5 * 1600 + 0.7 * 400]], dtype=F64))
    V = equation.total_velocity(V2_bar, halo**2)
    assert torch.allclose(V**2, V2_bar + halo**2)


def test_deep_mond_gives_baryonic_tully_fisher():
    M = 1e10
    a0 = tensors.A0_DEFAULT
    r = torch.tensor([300.0, 600.0], dtype=F64)  # kpc: g_N ≪ a₀
    g_bar = G * M / r**2
    V = torch.sqrt(equation.mond_acceleration(g_bar, a0) * r)
    assert torch.allclose(V**4, torch.full_like(V, G * M * a0), rtol=0.02)  # V⁴ = G M a₀, flat
    g_hi = torch.tensor([1e3 * a0], dtype=F64)
    assert float(equation.mond_acceleration(g_hi, a0) / g_hi) == pytest.approx(1.0, abs=1e-6)  # Newton when g ≫ a₀


def test_fit_recovers_synthetic_parameters():
    data, truth = tensors.synthetic_galaxies(CPU, noise=0.0)
    fit = solve.fit_all(data, iterations=600)
    assert torch.allclose(fit["nfw"]["ups_d"], truth["ups_d"], rtol=0.03)
    assert torch.allclose(fit["nfw"]["V_s"], truth["V_s"], rtol=0.03)
    assert torch.allclose(fit["nfw"]["r_s"], truth["r_s"], rtol=0.06)
    assert float(fit["nfw"]["chi2"].sum()) < 1e-2 * float(fit["baryons"]["chi2"].sum())


@pytest.fixture(scope="module")
def sparc():
    data = tensors.load_sparc(CPU)
    return data, solve.fit_all(data, iterations=500)


def test_sparc_data_loaded(sparc):
    data, _ = sparc
    assert data["names"] == ["NGC7814", "UGC02885", "NGC6503", "NGC2403", "DDO154", "NGC3198"]
    assert int(data["mask"].sum()) == 196
    assert bool((data["err"][data["mask"]] > 0).all())


def test_flat_curves_need_more_than_the_baryons(sparc):
    data, fit = sparc
    chi = {m: fit[m]["chi2_nu"] for m in ("baryons", "nfw", "burkert", "mond")}
    for m in ("nfw", "burkert"):
        assert bool((chi[m] < chi["baryons"]).all())  # a halo improves every galaxy
    assert float(chi["baryons"].median()) > 5  # with sensible M/L the baryons cannot reach the flat part
    assert float(chi["burkert"].median()) < 3
    a0 = float(fit["mond"]["a0"])
    assert 0.5 * tensors.A0_DEFAULT < a0 < 2.0 * tensors.A0_DEFAULT  # one acceleration scale near 1.2e-10 m/s²
