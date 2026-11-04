"""Checks written first (TDD): what any correct horizon-problem code must satisfy.

Run with `pytest problems/cosmology/variable_speed_of_light/test.py`. Each test is a cosmological law or a known number:
constant c gives back the standard ΛCDM horizon (closed form for matter + radiation, 14 Gpc today), the horizon problem
is quantified (thousands of causally disconnected patches on the CMB sky), and a large enough early c (VSL) or enough
e-folds of inflation make the whole last-scattering surface causally connected.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

F64 = torch.float64


@pytest.fixture(scope="module")
def grid():
    return tensors.log_a_grid(a_min=1e-62, n=40001, device=torch.device("cpu"))


def test_constant_c_and_no_inflation_recover_the_standard_horizon(grid):
    std = solve.horizons(grid, "lcdm")
    vsl = solve.horizons(grid, "vsl", c_ratio=torch.tensor([1.0], dtype=F64), a_star=1e-28)
    infl = solve.horizons(grid, "inflation", efolds=torch.tensor([0.0], dtype=F64), a_star=1e-28)
    assert torch.allclose(vsl["chi"][0], std["chi"][0], rtol=1e-12)
    assert torch.allclose(infl["chi"][0], std["chi"][0], rtol=1e-12)


def test_lcdm_horizon_matches_matter_radiation_closed_form(grid):
    std = solve.horizons(grid, "lcdm")
    a = torch.exp(grid)
    k = torch.searchsorted(a, torch.tensor([tensors.A_REC], dtype=F64)).item()
    a_eq = tensors.OMEGA_R / tensors.OMEGA_M
    # χ_p = (2c / H0√Ωm)(√(a + a_eq) − √a_eq) when Λ is negligible (it is, at recombination)
    exact = 2 * tensors.C0 / (tensors.H0 * math.sqrt(tensors.OMEGA_M)) * (math.sqrt(a[k] + a_eq) - math.sqrt(a_eq))
    assert std["chi"][0, k].item() == pytest.approx(exact, rel=1e-5)


def test_particle_horizon_today_is_about_14_gpc(grid):
    std = solve.horizons(grid, "lcdm")
    assert std["chi"][0, -1].item() / 1e3 == pytest.approx(14.2, rel=0.02)  # ≈ 46 billion light years


def test_horizon_problem_is_quantified(grid):
    std = solve.horizons(grid, "lcdm")
    stats = solve.sky_statistics(grid, std)
    # the particle horizon at recombination subtends only ~1–2° on today's sky …
    assert 1.0 < math.degrees(stats["theta"][0].item()) < 2.0
    # … so the CMB sky splits into ~10⁴ regions that never exchanged a signal, yet share T to 1 part in 10⁵
    assert stats["patches"][0].item() > 5e3
    assert not stats["solved"][0]


def test_vsl_horizon_matches_closed_form_and_solves_the_problem(grid):
    a_star, r = 1e-28, torch.tensor([1e30], dtype=F64)
    vsl = solve.horizons(grid, "vsl", c_ratio=r, a_star=a_star)
    std = solve.horizons(grid, "lcdm")
    extra = equation.vsl_extra_horizon(r, a_star)  # r c a_* /(H0 √Ωr) × πw / sin(πw) (radiation era, smooth step)
    k = torch.searchsorted(torch.exp(grid), torch.tensor([tensors.A_REC], dtype=F64)).item()
    assert (vsl["chi"][0, k] - std["chi"][0, k]).item() == pytest.approx(extra.item(), rel=1e-3)
    assert solve.sky_statistics(grid, vsl)["solved"][0]


def test_vsl_threshold_ratio(grid):
    a_star = 1e-28
    r_min = solve.vsl_minimum_ratio(grid, a_star)
    assert 1e25 < r_min < 1e28  # about 26 orders of magnitude for a transition at a_* = 1e-28
    r = torch.tensor([0.98 * r_min, 1.02 * r_min], dtype=F64)
    solved = solve.sky_statistics(grid, solve.horizons(grid, "vsl", c_ratio=r, a_star=a_star))["solved"]
    assert solved.tolist() == [False, True]


def test_inflation_needs_about_sixty_efolds(grid):
    a_star = 1e-28
    n_min = solve.inflation_minimum_efolds(grid, a_star)
    assert 55 < n_min < 70
    n = torch.tensor([n_min - 0.05, n_min + 0.05], dtype=F64)
    solved = solve.sky_statistics(grid, solve.horizons(grid, "inflation", efolds=n, a_star=a_star))["solved"]
    assert solved.tolist() == [False, True]


def test_comoving_hubble_radius_shrinks_during_inflation_and_at_the_vsl_transition(grid):
    a_star = 1e-28
    infl = solve.horizons(grid, "inflation", efolds=torch.tensor([60.0], dtype=F64), a_star=a_star)
    slope = torch.gradient(torch.log(infl["hubble_radius"][0]), spacing=(grid,))[0]
    during = (grid > math.log(a_star) - 55) & (grid < math.log(a_star) - 5)
    assert torch.allclose(slope[during], torch.tensor(-1.0, dtype=F64), atol=1e-6)  # c/(aH) ∝ 1/a in de Sitter
    std = solve.horizons(grid, "lcdm")
    slope = torch.gradient(torch.log(std["hubble_radius"][0]), spacing=(grid,))[0]
    rad = (grid > -40) & (grid < -20)
    assert torch.allclose(slope[rad], torch.tensor(1.0, dtype=F64), atol=1e-6)  # ∝ a in radiation era: grows
    vsl = solve.horizons(grid, "vsl", c_ratio=torch.tensor([1e30], dtype=F64), a_star=a_star)
    before = torch.searchsorted(grid, torch.tensor([math.log(a_star) - 5], dtype=F64)).item()
    after = torch.searchsorted(
        grid, torch.tensor([math.log(a_star) + 12], dtype=F64)
    ).item()  # step tail ∝ r (a_*/a)^(1/w)
    drop = vsl["hubble_radius"][0, before] / std["hubble_radius"][0, before]
    assert drop.item() == pytest.approx(1e30, rel=1e-6)  # c/(aH) is r times larger before the transition
    assert vsl["hubble_radius"][0, after].item() == pytest.approx(std["hubble_radius"][0, after].item(), rel=1e-12)


def test_vsl_scenario_respects_late_time_bounds(grid):
    c = equation.speed_of_light(torch.exp(grid), torch.tensor([1e30], dtype=F64), 1e-28)  # (1, A)
    late = torch.exp(grid) > tensors.A_REC / 10
    # constant today and since before recombination: Δc/c = 0 within the CMB (~10⁻³) and GW170817 (~10⁻¹⁵) bounds
    assert ((c[0, late] / tensors.C0 - 1).abs().max() < 1e-15).item()
    bounds = equation.BOUNDS
    assert bounds["GW170817 speed of gravity"]["value"][0] < 0 < bounds["GW170817 speed of gravity"]["value"][1]
