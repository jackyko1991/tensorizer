"""Physics checks, written first (TDD): what any correct Kerr–Newman code must satisfy.

Run with `pytest problems/astro/kerr_newman_ergosphere/test.py`. Each test is a known closed form or limit:
Schwarzschild (a = Q = 0) has r₊ = 2M, ISCO 6M and photon sphere 3M; Reissner–Nordström (a = 0) has
r± = M ± √(M² − Q²) and an extremal ISCO at 4M; Kerr (Q = 0) matches the Bardeen–Press–Teukolsky ISCO and the
photon-orbit formula, with the prograde ISCO → M at extremality; the ergosphere touches the horizon at the poles;
frame dragging equals the horizon angular velocity at r₊; beyond a² + Q² = M² there is no horizon.
"""

import math

import pytest
import torch

from . import equation, tensors

F64 = torch.float64


def t(*v):
    return torch.tensor(v, dtype=F64)


def test_schwarzschild_limit():
    M, a, Q = t(1.0, 2.5), t(0.0, 0.0), t(0.0, 0.0)
    r_plus, r_minus = equation.horizons(M, a, Q)
    assert torch.allclose(r_plus, 2 * M) and torch.allclose(r_minus, torch.zeros(2, dtype=F64))
    orb = equation.equatorial_orbits(M, a, Q)
    for key in ("isco_pro", "isco_retro"):
        assert torch.allclose(orb[key], 6 * M, rtol=1e-4)  # ISCO = 6M, no spin ⇒ no pro/retro difference
    for key in ("photon_pro", "photon_retro"):
        assert torch.allclose(orb[key], 3 * M, rtol=1e-4)  # photon sphere = 3M


def test_horizons_scale_with_mass():
    M = t(0.5, 1.0, 3.0)
    a, Q = 0.6 * M, 0.3 * M
    r_plus, r_minus = equation.horizons(M, a, Q)
    assert torch.allclose(r_plus / M, r_plus[1].expand(3))  # only a/M and Q/M matter (G = c = 1)
    assert torch.allclose(r_plus * r_minus, a**2 + Q**2)  # Vieta: roots of Δ = r² − 2Mr + a² + Q²
    assert torch.allclose(r_plus + r_minus, 2 * M)


def test_metric_identity_and_delta():
    # Kerr–Newman in Boyer–Lindquist coordinates: g_tt g_φφ − g_tφ² = −Δ sin²θ
    M, a, Q = t(1.0, 1.0, 1.0), t(0.0, 0.7, 0.5), t(0.4, 0.0, 0.6)
    r, th = t(2.7, 3.1, 4.4), t(0.4, 1.2, 2.0)
    g_tt, g_tp, g_pp = equation.metric_t_phi(r, th, M, a, Q)
    delta = r**2 - 2 * M * r + a**2 + Q**2
    assert torch.allclose(g_tt * g_pp - g_tp**2, -delta * torch.sin(th) ** 2, rtol=1e-12)


def test_reissner_nordstrom_limit():
    M, Q = t(1.0, 1.0, 1.0), t(0.5, 0.9, 1.0)
    a = torch.zeros(3, dtype=F64)
    r_plus, _ = equation.horizons(M, a, Q)
    assert torch.allclose(r_plus, M + torch.sqrt(M**2 - Q**2))
    orb = equation.equatorial_orbits(M, a, Q)
    rn_photon = (3 * M + torch.sqrt(9 * M**2 - 8 * Q**2)) / 2  # RN photon sphere
    assert torch.allclose(orb["photon_pro"], rn_photon, rtol=1e-4)
    assert orb["isco_pro"][2].item() == pytest.approx(4.0, rel=1e-3)  # extremal RN: ISCO = 4M
    # ergosphere = horizon when a = 0 (no frame dragging, no ergoregion)
    th = torch.linspace(0, math.pi, 7, dtype=F64)
    r_E = equation.ergosphere(th[None, :], M[:, None], a[:, None], Q[:, None])
    assert torch.allclose(r_E, r_plus[:, None].expand_as(r_E))


def test_kerr_isco_and_photon_orbits_match_bardeen():
    a = t(0.3, 0.7, 0.95)
    M, Q = torch.ones(3, dtype=F64), torch.zeros(3, dtype=F64)
    orb = equation.equatorial_orbits(M, a, Q)
    assert torch.allclose(orb["isco_pro"], equation.kerr_isco(M, a, prograde=True), rtol=2e-4)
    assert torch.allclose(orb["isco_retro"], equation.kerr_isco(M, a, prograde=False), rtol=2e-4)
    ph_pro = 2 * M * (1 + torch.cos(2 / 3 * torch.acos(-a / M)))  # BPT photon orbits
    ph_retro = 2 * M * (1 + torch.cos(2 / 3 * torch.acos(a / M)))
    assert torch.allclose(orb["photon_pro"], ph_pro, rtol=1e-4)
    assert torch.allclose(orb["photon_retro"], ph_retro, rtol=1e-4)
    assert (orb["isco_pro"] < orb["isco_retro"]).all()  # co-rotating orbits get closer


def test_extremal_kerr_prograde_isco_goes_to_M():
    one = torch.ones(1, dtype=F64)
    assert equation.kerr_isco(one, one, prograde=True).item() == pytest.approx(1.0, abs=1e-12)  # → M
    assert equation.kerr_isco(one, one, prograde=False).item() == pytest.approx(9.0, abs=1e-12)  # retro → 9M
    a = t(0.99, 0.999, 0.9999)
    orb = equation.equatorial_orbits(torch.ones(3, dtype=F64), a, torch.zeros(3, dtype=F64))
    r = orb["isco_pro"]
    assert (r.diff() < 0).all() and r[-1] < 1.15  # approaches M as a → M (slowly, ∝ (1 − a)^{1/3})
    assert torch.allclose(r, equation.kerr_isco(torch.ones(3, dtype=F64), a, prograde=True), rtol=2e-3)


def test_ergosphere_touches_horizon_at_poles_and_bulges_at_equator():
    M, a, Q = t(1.0, 1.0, 2.0), t(0.5, 0.9, 1.0), t(0.3, 0.0, 0.8)
    r_plus, _ = equation.horizons(M, a, Q)
    th = torch.tensor([0.0, math.pi / 2, math.pi], dtype=F64)
    r_E = equation.ergosphere(th[None, :], M[:, None], a[:, None], Q[:, None])  # (B, 3)
    assert torch.allclose(r_E[:, 0], r_plus) and torch.allclose(r_E[:, 2], r_plus)
    assert torch.allclose(r_E[:, 1], M + torch.sqrt(M**2 - Q**2))  # equator: M + √(M² − Q²)
    assert (r_E[:, 1] > r_plus).all()
    # g_tt = 0 on the ergosurface (static observers impossible inside)
    g_tt, _, _ = equation.metric_t_phi(r_E[:, 1], th[1].expand(3), M, a, Q)
    assert g_tt.abs().max() < 1e-12


def test_frame_dragging():
    M, a, Q = t(1.0, 1.0), t(0.6, 0.9), t(0.4, 0.2)
    r_plus, _ = equation.horizons(M, a, Q)
    th = t(1.0, 0.5)
    omega_h = equation.frame_dragging(r_plus, th, M, a, Q)
    assert torch.allclose(omega_h, a / (r_plus**2 + a**2))  # rigid rotation of the horizon Ω_H, any θ
    r = t(1e3, 1e3)
    omega_far = equation.frame_dragging(r, t(math.pi / 2, math.pi / 2), M, a, Q)
    assert torch.allclose(omega_far, 2 * a * M / r**3, rtol=1e-2)  # Lense–Thirring far field


def test_naked_singularity_regime():
    M, a, Q = t(1.0, 1.0, 1.0), t(0.9, 0.8, 0.6), t(0.6, 0.6, 0.8)  # a² + Q² = 1.17, 1.0, 1.0
    r_plus, _ = equation.horizons(M, a, Q)
    naked = equation.is_naked(M, a, Q)
    assert naked.tolist() == [True, False, False]
    assert torch.isnan(r_plus[0]) and torch.allclose(r_plus[1:], M[1:])  # extremal: r₊ = r₋ = M
    kappa = equation.surface_gravity(M, a, Q)
    assert kappa[1:].abs().max() < 1e-12  # extremal holes are cold (κ = 0)


def test_batched_grid():
    M, a, Q = tensors.parameter_grid(n=9, masses=(1.0, 3.0))
    assert M.shape == a.shape == Q.shape and M.ndim == 3
    naked = equation.is_naked(M, a, Q)
    assert torch.equal(naked, a**2 + Q**2 > M**2 * (1 + 1e-12))  # extremal points count as holes
    r_plus, _ = equation.horizons(M, a, Q)
    assert torch.isnan(r_plus[naked]).all() and (r_plus[~naked] >= M[~naked] - 1e-12).all()
