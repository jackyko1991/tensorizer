"""Physics checks, written first (TDD): what Einstein's two 1905 relativity papers predict.

Run with `pytest problems/astro/einstein_1905_relativity/test.py`. From the two postulates: a light clock moving at
speed v ticks γ times slower, a moving rod is 1/γ shorter, simultaneity is relative, speeds never add beyond c and
the interval is invariant; from the mass–energy paper: a body that emits energy L loses mass L/c², which the He-4
mass defect and e⁺e⁻ annihilation confirm. Frisch & Smith's muons show the dilation γ ≈ 9.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

f64 = torch.float64


def test_gamma_tends_to_one_at_low_speed():
    beta = torch.tensor([1e-6, 1e-4, 1e-2], dtype=f64)
    g = equation.gamma(beta)
    assert torch.allclose(g - 1, beta**2 / 2, rtol=1e-3)  # γ ≈ 1 + β²/2: Newton's world
    assert (equation.gamma(torch.tensor([0.9, 0.99, 0.999], dtype=f64)).diff() > 0).all()


def test_light_clock_ticks_slower_by_gamma():
    beta = tensors.clock_speeds(0.95, 5)  # (B,) including β = 0
    t, x, y = solve.light_clocks(beta, D=1.0, t_max=12.0, n=4001)  # photon positions (B, T)
    ticks = solve.count_ticks(y)  # completed round trips per clock (B,)
    # the photon travels the diagonal at c, so a moving clock ticks γ times less often: one tick per 2Dγ/c
    expected = torch.floor(12.0 / (2 * 1.0 * equation.gamma(beta)))
    assert torch.allclose(ticks.double(), expected, atol=1)
    assert ticks[0] >= 5 * ticks[-1]  # β = 0.95: γ ≈ 3.2
    # every photon moves at c in the lab, |dr/dt| = 1 (second postulate); chords across a bounce are only shorter
    speed = torch.hypot(x.diff(dim=1), y.diff(dim=1)) / t.diff()
    assert (speed <= 1 + 1e-9).all()
    assert ((speed - 1).abs() < 1e-9).double().mean() > 0.98


def test_length_contraction_from_light_round_trips():
    beta = torch.tensor([0.0, 0.3, 0.6, 0.9, 0.99], dtype=f64)
    L = solve.contracted_length(L0=1.0, beta=beta)  # rod length that keeps along- and across-clocks in step
    assert torch.allclose(L, 1 / equation.gamma(beta), rtol=1e-12)


def test_relativity_of_simultaneity():
    beta = torch.tensor([0.2, 0.5, 0.9], dtype=f64)
    events = torch.tensor([[0.0, -1.0, 0, 0], [0.0, 1.0, 0, 0]], dtype=f64)  # two flashes at t = 0, x = ∓1
    boosted = equation.transform(equation.boost(beta), events)  # (B, 2, 4)
    dt = boosted[:, 1, 0] - boosted[:, 0, 0]
    assert torch.allclose(dt, -equation.gamma(beta) * beta * 2.0)  # Δt' = −γβΔx/c²: not simultaneous for the train


def test_velocity_addition_never_exceeds_c():
    u = torch.linspace(-0.999, 0.999, 201, dtype=f64)
    w = equation.velocity_addition(u[:, None], u[None, :])  # (U, V) grid
    assert w.abs().max() < 1
    assert torch.allclose(equation.velocity_addition(torch.tensor(1.0, dtype=f64), u), torch.ones_like(u))  # c + v = c
    # composing two boosts is one boost with the added velocity
    b1, b2 = torch.tensor(0.6, dtype=f64), torch.tensor(0.7, dtype=f64)
    combined = equation.boost(b1) @ equation.boost(b2)
    assert torch.allclose(combined, equation.boost(equation.velocity_addition(b1, b2)), atol=1e-12)


def test_interval_is_invariant():
    g = torch.Generator().manual_seed(0)
    events = torch.randn(100, 4, generator=g, dtype=f64)
    beta = torch.tensor([-0.9, 0.1, 0.5, 0.99], dtype=f64)
    boosted = equation.transform(equation.boost(beta), events)  # (B, N, 4)
    s2 = equation.interval(events)
    assert torch.allclose(equation.interval(boosted), s2.expand_as(boosted[..., 0]), atol=1e-9)


def test_energy_momentum_relation():
    m = torch.tensor([0.511, 105.66, 938.27], dtype=f64)  # e, μ, p (MeV)
    beta = torch.tensor([0.0, 0.5, 0.99], dtype=f64)
    p4 = equation.four_momentum(m[:, None], beta[None, :])  # (3, 3, 4): (E, pc, 0, 0)
    E, p = p4[..., 0], p4[..., 1]
    assert torch.allclose(E**2, p**2 + m[:, None] ** 2, rtol=1e-12)  # E² = (pc)² + (mc²)²
    assert torch.allclose(equation.interval(p4), -(m[:, None] ** 2) * torch.ones_like(E))


def test_einstein_1905_mass_loss_is_L_over_c2():
    beta = torch.linspace(0.01, 0.99, 50, dtype=f64)
    theta = torch.linspace(0, math.pi, 7, dtype=f64)
    L = 1.0
    dK = solve.kinetic_energy_drop(L, beta[:, None], theta[None, :])  # (B, Θ): K0 − K1 seen from the moving frame
    # K0 − K1 = (L/c²)(γ − 1)c²: exactly the kinetic energy of a mass L/c², at every speed and emission angle
    dm = dK / (equation.gamma(beta)[:, None] - 1)
    assert torch.allclose(dm, torch.full_like(dm, L), rtol=1e-10)
    # Einstein's low-speed form: K0 − K1 ≈ ½ (L/c²) v²
    assert dK[0, 0].item() == pytest.approx(0.5 * L * beta[0].item() ** 2, rel=1e-3)


def test_helium_mass_defect():
    B = equation.binding_energy(*tensors.nuclear_masses(["He-4"]))
    assert B.item() == pytest.approx(28.296, abs=0.01)  # MeV: 0.0304 u of mass missing from 2p + 2n


def test_electron_positron_annihilation():
    m = tensors.M_ELECTRON
    photons = solve.annihilate_at_rest(m)  # (2, 4) back-to-back photon four-momenta
    assert torch.allclose(photons[:, 0], torch.full((2,), m, dtype=f64))  # 511 keV each
    assert torch.allclose(photons.sum(0)[1:], torch.zeros(3, dtype=f64))  # momentum conserved
    assert torch.allclose(equation.interval(photons), torch.zeros(2, dtype=f64), atol=1e-12)  # massless


def test_muons_need_time_dilation():
    data = tensors.frisch_smith()
    r = solve.muon_gamma(data, dEdx=1.82)
    assert r["no_dilation_rate"] < 40  # without dilation only ~30 muons/h would survive the trip
    assert (data["rate"][1] - r["no_dilation_rate"]) / data["err"][1] > 30  # excluded by > 30 σ
    # γ inferred from the counts agrees with γ from the measured speeds within two standard errors
    assert abs(r["gamma_fit"] - r["gamma_pred"]) < 2 * r["gamma_err"]
    assert 7.5 < r["gamma_fit"] < 10.5
