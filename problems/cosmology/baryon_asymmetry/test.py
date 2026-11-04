"""Checks written first (TDD): what any correct leptogenesis Boltzmann solver must satisfy.

Run with `pytest problems/cosmology/baryon_asymmetry/test.py`. Two of Sakharov's conditions appear as tests (no
CP violation, or no departure from equilibrium, ⇒ no asymmetry); the weak-washout limit and the integrals of the
rates have closed forms; the asymmetry is linear and increasing in ε and is washed out at large K.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def run(eps, K, **kw):
    eps_t = torch.as_tensor(eps, dtype=torch.float64).reshape(-1, 1)  # (E, 1)
    K_t = torch.as_tensor(K, dtype=torch.float64).reshape(1, -1)  # (1, K)
    z = tensors.z_grid(kw.pop("z_max", 100.0), kw.pop("n", 4000), CPU)
    return solve.solve(z, eps_t, K_t, **kw)


def test_equilibrium_abundance_and_its_derivative():
    z = torch.tensor([1e-3, 0.5, 1.0, 3.0, 10.0], dtype=torch.float64)
    assert equation.n_eq(z)[0].item() == pytest.approx(0.75, rel=1e-5)  # relativistic: 3/4 of the photons
    h = 1e-6
    numeric = (equation.n_eq(z + h) - equation.n_eq(z - h)) / (2 * h)
    assert torch.allclose(equation.dn_eq_dz(z), numeric, rtol=1e-6)  # d(z²K₂)/dz = −z²K₁


def test_rates_have_the_textbook_integrals():
    # ∫₀^∞ W dz = (K/4) ∫ z³K₁ dz = (K/4)(3π/2), and D → K z at late times (time dilation gone)
    z = tensors.z_grid(200.0, 200000, CPU)
    W = equation.washout(z, torch.tensor(1.0, dtype=torch.float64))
    assert torch.trapezoid(W, z).item() == pytest.approx(3 * math.pi / 8, rel=1e-4)
    D = equation.decay_rate(torch.tensor([50.0], dtype=torch.float64), torch.tensor(2.0, dtype=torch.float64))
    assert D.item() == pytest.approx(2.0 * 50.0 * (1 - 1.5 / 50), rel=2e-3)  # K₁/K₂ ≈ 1 − 3/(2z)


def test_no_cp_violation_means_no_asymmetry():
    sol = run([0.0], [0.1, 1.0, 10.0, 100.0])
    assert sol["n_bl"].abs().max().item() == 0.0  # Sakharov: C and CP must be violated


def test_no_departure_from_equilibrium_means_no_asymmetry():
    sol = run([1e-6], [0.1, 1.0, 10.0], equilibrium=True)
    assert sol["n_bl"].abs().max().item() == 0.0  # Sakharov: out of equilibrium, or nothing survives


def test_weak_washout_thermal_limit_gives_full_efficiency():
    # K ≪ 1, thermal start: every one of the 3/4 neutrinos per photon decays with almost no washout ⇒ κ_f → 1
    sol = run([1e-6], [0.01, 0.03], z_max=150.0, n=6000)
    kappa = solve.efficiency(sol["n_bl_final"], torch.tensor([[1e-6]], dtype=torch.float64))
    assert kappa[0, 0].item() == pytest.approx(1.0, abs=0.02)
    assert 0.95 < kappa[0, 1].item() < kappa[0, 0].item()  # washout grows with K
    assert sol["n_n"][..., -1].abs().max() < 1e-8  # all heavy neutrinos have decayed


def test_strong_washout_follows_the_analytic_fit():
    # Buchmüller, Di Bari, Plümacher (2005): κ_f ≈ (2 / (z_B K)) (1 − e^{−z_B K / 2}),  z_B = 2 + 4 K^0.13 e^{−2.5/K}
    Ks = torch.tensor([10.0, 30.0, 100.0], dtype=torch.float64)
    sol = run([1e-6], Ks.tolist())
    kappa = solve.efficiency(sol["n_bl_final"], torch.tensor([[1e-6]], dtype=torch.float64))[0]
    fit = equation.kappa_strong_washout(Ks)
    assert torch.allclose(kappa, fit, rtol=0.15)  # the fit is good to ~10%
    assert (kappa.diff() < 0).all()  # more washout, less asymmetry


def test_asymmetry_is_linear_and_increasing_in_epsilon():
    eps = [1e-8, 1e-7, 3e-7, 1e-6, 1e-5]
    sol = run(eps, [0.5, 5.0, 50.0], initial="zero")
    eta = solve.baryon_to_photon(sol["n_bl_final"])  # (E, K)
    assert (eta.diff(dim=0) > 0).all()  # monotonic in ε
    ratio = eta / torch.tensor(eps, dtype=torch.float64)[:, None]
    assert torch.allclose(ratio, ratio[:1].expand_as(ratio), rtol=1e-10)  # the source is linear in ε


def test_step_size_convergence():
    a = run([1e-6], [0.3, 3.0, 30.0], n=2000)["n_bl_final"]
    b = run([1e-6], [0.3, 3.0, 30.0], n=8000)["n_bl_final"]
    assert torch.allclose(a, b, rtol=2e-3)


def test_observed_asymmetry_is_reachable():
    # η_B ≈ 6.1e-10 needs ε ≈ 6.1e-10 / (0.96e-2 κ_f): about 1e-6 for K ≈ 10
    sol = run(torch.logspace(-8, -4, 41).tolist(), [10.0])
    eta = solve.baryon_to_photon(sol["n_bl_final"])[:, 0]
    assert eta.min() < tensors.ETA_OBSERVED < eta.max()
