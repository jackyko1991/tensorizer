"""Physics checks, written first (TDD): what any correct orbit-transfer code must satisfy.

Run with `pytest problems/astro/orbit_transfer/test.py`. Each test is a physical law, not an implementation detail:
the Δv and transfer times match the Hohmann and bi-elliptic closed forms, the bi-elliptic crossover lies at r2/r1 ≈
11.94 and 15.58, and energy is conserved between burns.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")

EARTH = tensors.BODIES["earth"]
F64 = torch.float64


def _t(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=F64)


@pytest.mark.parametrize("r1, r2", [(1.0, 6.3), (1.0, 1.524), (5.0, 2.0), (1.0, 40.0)])
def test_hohmann_matches_closed_form(r1, r2):
    """Δv₁ = √(μ/r₁)(√(2r₂/(r₁+r₂)) − 1), Δv₂ = √(μ/r₂)(1 − √(2r₁/(r₁+r₂))), t = π√(((r₁+r₂)/2)³/μ)."""
    mu = 1.0
    dv1, dv2, t = equation.hohmann(_t(r1), _t(r2), mu)
    assert float(dv1) == pytest.approx(math.sqrt(mu / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1), rel=1e-12)
    assert float(dv2) == pytest.approx(math.sqrt(mu / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2))), rel=1e-12)
    assert float(t) == pytest.approx(math.pi * math.sqrt(((r1 + r2) / 2) ** 3 / mu), rel=1e-12)


def test_leo_to_geo():
    """Textbook LEO (≈ 300 km) → GEO costs about 3.9 km/s in about 5.3 h."""
    dv1, dv2, t = equation.hohmann(_t(6678.0), _t(42164.0), EARTH["mu"])
    assert float(dv1 + dv2) == pytest.approx(3.89, abs=0.02)
    assert float(t) / 3600 == pytest.approx(5.27, abs=0.05)


def test_bi_elliptic_reduces_to_hohmann_when_rb_equals_r2():
    h1, h2, th = equation.hohmann(_t(1.0), _t(8.0), 1.0)
    b1, b2, b3, tb = equation.bi_elliptic(_t(1.0), _t(8.0), _t(8.0), 1.0)
    assert float(b1) == pytest.approx(float(h1), rel=1e-12)
    assert float(b2) == pytest.approx(float(h2), rel=1e-12)
    assert abs(float(b3)) < 1e-12
    # with r_b = r2 the second "ellipse" is the target circle itself: half of its period is added
    assert float(tb) == pytest.approx(float(th) + math.pi * math.sqrt(8.0**3), rel=1e-12)


def test_bi_elliptic_crossovers():
    assert solve.crossover_any_rb() == pytest.approx(11.94, abs=0.01)
    assert solve.crossover_every_rb() == pytest.approx(15.58, abs=0.01)


def test_crossover_seen_on_the_batched_curves():
    R = tensors.ratio_grid(2000)
    c = solve.total_dv_curves(R, rb_ratio=2.0)
    cheaper = c["bi-parabolic"] < c["hohmann"]
    first = float(R[cheaper.nonzero()[0]])
    assert first == pytest.approx(11.94, rel=0.01)
    assert not cheaper[R < 11.9].any() and cheaper[R > 12.0].all()
    # above 15.58 every intermediate radius wins, below 11.94 none does
    rb = torch.logspace(0.001, 3, 500, dtype=F64)  # r_b / r2 from just above 1 to 1000
    for ratio, expect_all, expect_any in [(16.0, True, True), (11.5, False, False)]:
        one, r2 = torch.ones_like(rb), torch.full_like(rb, ratio)
        b1, b2, b3, _ = equation.bi_elliptic(one, r2, rb * ratio, 1.0)
        h1, h2, _ = equation.hohmann(one, r2, 1.0)
        wins = b1.abs() + b2.abs() + b3.abs() < h1.abs() + h2.abs()
        assert bool(wins.all()) == expect_all and bool(wins.any()) == expect_any


def test_earth_mars_hohmann():
    sun = tensors.BODIES["sun"]
    au = _t(sun["unit_km"])
    dv1, dv2, t = equation.hohmann(au, tensors.MARS_AU * au, sun["mu"])
    assert 2.9 < float(dv1) < 3.0
    assert 2.6 < float(dv2) < 2.7
    assert 5.5 < float(dv1 + dv2) < 5.7  # ≈ 5.6 km/s heliocentric
    assert 255 < float(t) / 86400 < 262  # ≈ 259 days


@pytest.fixture(scope="module")
def flight():
    """LEO → GEO with both transfers in one batch, integrated finely in float64."""
    mu = EARTH["mu"]
    r1, r2, rb = 1.05 * EARTH["unit_km"], 6.61 * EARTH["unit_km"], 2 * 6.61 * EARTH["unit_km"]
    kinds = ["hohmann", "bi-elliptic"]
    y0 = tensors.initial_state(r1, 2, mu, CPU)
    burn_t, burn_dv = tensors.burn_schedule(kinds, r1, r2, rb, mu, CPU)
    T1 = 2 * math.pi * math.sqrt(r1**3 / mu)
    t_end = float(burn_t.max()) + 2 * math.pi * math.sqrt(r2**3 / mu)  # one full target orbit after arrival
    times, traj, burn_pos = solve.fly(y0, mu, burn_t, burn_dv, t_end, T1 / 300)
    return mu, r1, r2, rb, burn_t, times, traj, burn_pos


def test_arrives_on_target_orbit_at_the_right_time(flight):
    mu, r1, r2, rb, burn_t, times, traj, burn_pos = flight
    r = traj[:, :, 0, :].norm(dim=-1)  # (T, B)
    _, _, t_h = equation.hohmann(_t(r1), _t(r2), mu)
    *_, t_bi = equation.bi_elliptic(_t(r1), _t(r2), _t(rb), mu)
    for b, t_arrive in enumerate([float(t_h), float(t_bi)]):
        k = int(torch.searchsorted(times, torch.tensor(t_arrive, dtype=F64)))
        assert float(times[k]) == pytest.approx(t_arrive, rel=1e-12)
        assert float(r[k, b]) == pytest.approx(r2, rel=1e-6)  # at r2 exactly when the formula says
        if b == 0:
            assert float(r[:k, b].max()) < r2 * (1 + 1e-6)  # the Hohmann ellipse just touches r2
        after = times > t_arrive
        assert float((r[after, b] - r2).abs().max()) / r2 < 1e-6  # and stays on the circle after the last burn
    # the burns fire at the apsides: on the +x axis at r1, then at -x (apoapsis)
    assert float(burn_pos[0, 1, 0]) == pytest.approx(-r2, rel=1e-6)
    assert float(burn_pos[1, 1, 0]) == pytest.approx(-rb, rel=1e-6)
    assert float(burn_pos[1, 2, 0]) == pytest.approx(r2, rel=1e-6)


def test_energy_conserved_between_burns(flight):
    mu, r1, r2, rb, burn_t, times, traj, _ = flight
    E = equation.energy(traj, mu)  # (T, B)
    for b in range(2):
        edges = sorted({float(t) for t in burn_t[b]} | {float(times[-1]) + 1})
        for t0, t1 in zip(edges[:-1], edges[1:]):
            arc = (times >= t0) & (times < t1)  # at a burn time the stored state is the one after the burn
            assert (E[arc, b] / E[arc, b][0] - 1).abs().max() < 1e-7  # RK4 drift only
    # and each arc has the energy -μ/2a of the orbit it should be on
    t_h = float(burn_t[0, 1])
    arc = (times > 0) & (times < t_h)
    assert float(E[arc, 0].mean()) == pytest.approx(-mu / (r1 + r2), rel=1e-8)  # vis-viva: E = −μ/2a on the ellipse
