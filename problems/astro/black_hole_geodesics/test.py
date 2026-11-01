"""Physics checks, written first (TDD): what any correct Schwarzschild-geodesic code must satisfy.

Run with `pytest problems/astro/black_hole_geodesics/test.py`. Each test is a physical law, not an implementation
detail: the Christoffel symbols match Schwarzschild, the four-velocity stays normalised, the weak-field perihelion
advance is 6πM/p, the ISCO is 6M, the photon sphere 3M and infall takes finite proper time.
"""

import math

import pytest
import torch

from . import equation, solve, tensors


def test_autograd_christoffels_match_schwarzschild():
    M, r, th = 1.0, 7.3, 1.1
    x = torch.tensor([[0.4, r, th, 0.9]], dtype=torch.float64)
    G = equation.christoffel(x, M)[0]
    expected = torch.zeros(4, 4, 4, dtype=torch.float64)
    f = 1 - 2 * M / r
    expected[0, 0, 1] = expected[0, 1, 0] = M / (r**2 * f)
    expected[1, 0, 0] = M * f / r**2
    expected[1, 1, 1] = -M / (r**2 * f)
    expected[1, 2, 2] = -r * f
    expected[1, 3, 3] = -r * f * math.sin(th) ** 2
    expected[2, 1, 2] = expected[2, 2, 1] = 1 / r
    expected[2, 3, 3] = -math.sin(th) * math.cos(th)
    expected[3, 1, 3] = expected[3, 3, 1] = 1 / r
    expected[3, 2, 3] = expected[3, 3, 2] = math.cos(th) / math.sin(th)
    assert torch.allclose(G, expected, atol=1e-12)


def test_geodesic_keeps_four_velocity_normalised():
    _, traj = solve.geodesic(tensors.bound_orbit(14.0, 0.4), 1.0, 600.0, 2.0)
    assert (equation.norm(traj, 1.0) + 1).abs().max() < 1e-6  # timelike geodesic: g_μν u^μ u^ν = −1


def test_weak_field_perihelion_precession():
    p, e = 100.0, 0.3
    a = p / (1 - e**2)
    _, traj = solve.geodesic(tensors.bound_orbit(p, e), 1.0, 1.3 * 2 * math.pi * a**1.5, 20.0)
    advance = solve.periapsis_angles(traj).diff()[0] - 2 * math.pi
    # GR perihelion advance 6πM/p
    assert advance.item() == pytest.approx(equation.weak_field_precession(p, 1.0), rel=0.1)


def test_isco_and_photon_sphere_radii():
    M = 1.0
    r = torch.linspace(3.5, 20, 20001, dtype=torch.float64)
    # at L² = 12 M² the minimum and maximum of V merge: dV/dr and d²V/dr² vanish together at r = 6M
    V = equation.effective_potential(r, math.sqrt(12) * M, M)
    dV = torch.gradient(V, spacing=(r,))[0]
    assert r[dV.abs().argmin()].item() == pytest.approx(equation.isco_radius(M), rel=1e-2)
    # light: V_light ∝ (1 − 2M/r)/r² peaks at the photon sphere
    rr = torch.linspace(2.1, 10, 20001, dtype=torch.float64)
    V_light = (1 - 2 * M / rr) / rr**2
    assert rr[V_light.argmax()].item() == pytest.approx(equation.photon_sphere_radius(M), rel=1e-3)


def test_radial_infall_reaches_horizon_in_finite_proper_time():
    r0 = 10.0
    tau, traj = solve.infall(tensors.radial_infall(r0), 1.0, 0.1)
    r, t = traj[:, 0, 0, 1], traj[:, 0, 0, 0]
    assert r[-1].item() == pytest.approx(2.1, abs=0.02)
    assert tau[-1].item() < solve.proper_infall_time(r0, 2.0, 1.0) < 40  # finite
    assert t[-1] > tau[-1]  # the far-away clock runs ahead: t → ∞ at the horizon
    assert solve.proper_infall_time(r0, 0.0, 1.0) == pytest.approx(math.pi / 2 * math.sqrt(r0**3 / 2))
