"""Physics checks, written first (TDD): what any correct orbit fit around Sgr A* must satisfy.

Run with `pytest problems/astro/sgr_a_star_orbits/test.py`. Each test is a law, not an implementation detail:
Kepler's equation is solved to machine precision, a circular orbit has constant speed, the period obeys Kepler's
third law, energy and angular momentum are conserved along the orbit, the precession formula gives Mercury's 43″
per century, autograd derivatives agree with finite differences, and the batched fit recovers known elements.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

# Gillessen et al. 2017 (S2 only): a, e, i, Ω, ω, t_p, and the potential M = 4.35e6 M☉, R0 = 8.33 kpc
PUBLISHED = {
    "a": 125.5,
    "e": 0.8839,
    "inc": 134.18,
    "Omega": 226.94,
    "omega": 65.51,
    "tp": 2002.33,
    "M": 4.35,
    "R0": 8.33,
}


def elements(**over) -> torch.Tensor:
    p = {**PUBLISHED, **over}
    return tensors.pack(p)[None]  # (1, 8)


def test_kepler_equation_is_solved_to_machine_precision():
    e = torch.linspace(0.0, 0.99, 100, dtype=torch.float64)[:, None]
    mean = torch.linspace(-3 * math.pi, 3 * math.pi, 2001, dtype=torch.float64)[None]
    big_e = equation.kepler_solve(mean, e)  # (100, 2001) all at once
    assert (big_e - e * torch.sin(big_e) - mean).abs().max().item() < 1e-12
    # the implicit-function derivative dE/dM = 1/(1 − e cos E) comes out of autograd
    m = mean.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(equation.kepler_solve(m, e).sum(), m)
    assert torch.allclose(grad[0], (1 / (1 - e * torch.cos(big_e))).sum(0), rtol=1e-10)  # summed over the e rows


def test_circular_orbit_has_constant_speed_and_radius():
    t = torch.linspace(1990.0, 2030.0, 500, dtype=torch.float64)
    r, v = equation.state_vectors(elements(e=0.0), t)  # (1, T, 3) AU and AU/yr
    speed, radius = v.norm(dim=-1), r.norm(dim=-1)
    assert (speed.std() / speed.mean()).item() < 1e-12
    assert (radius.std() / radius.mean()).item() < 1e-12
    gm = equation.gm_au3_yr2(torch.tensor(4.35e6, dtype=torch.float64))
    assert speed.mean().item() == pytest.approx(math.sqrt(gm.item() / radius.mean().item()), rel=1e-12)  # v² = GM/r


def test_period_obeys_keplers_third_law():
    p = elements()
    period = equation.period(p)  # years, from P² = a³/(M/M☉) in AU and yr
    a_au = PUBLISHED["a"] / 1000 * PUBLISHED["R0"] * 1000
    assert period.item() == pytest.approx(math.sqrt(a_au**3 / 4.35e6), rel=1e-3)
    assert 15.5 < period.item() < 16.5  # S2: about 16 yr
    t = torch.linspace(1995.0, 2010.0, 40, dtype=torch.float64)
    sky0 = equation.observables(p, t)
    sky1 = equation.observables(p, t + period)
    for a, b in zip(sky0, sky1):
        assert torch.allclose(a, b, atol=1e-8)  # the orbit closes after one period
    # the SI constants give 4π²/(G M☉) ≈ 1 yr²/AU³ (Kepler's law in astronomers' units)
    assert equation.gm_au3_yr2(torch.tensor(1.0, dtype=torch.float64)).item() == pytest.approx(4 * math.pi**2, rel=1e-3)


def test_energy_and_angular_momentum_are_conserved():
    t = torch.linspace(1995.0, 2012.0, 4000, dtype=torch.float64)
    p = elements()
    r, v = equation.state_vectors(p, t)
    gm = equation.gm_au3_yr2(torch.tensor(4.35e6, dtype=torch.float64))
    energy = 0.5 * v.pow(2).sum(-1) - gm / r.norm(dim=-1)  # (1, T)
    ang = torch.linalg.cross(r, v)  # (1, T, 3)
    assert (energy.std() / energy.mean().abs()).item() < 1e-10
    assert (ang - ang[:, :1]).abs().max().item() < 1e-8 * ang.norm(dim=-1).max().item()
    a_au = PUBLISHED["a"] / 1000 * PUBLISHED["R0"] * 1000  # mas → arcsec, kpc → pc: 1″ at 1 pc is 1 AU
    assert energy.mean().item() == pytest.approx(-gm.item() / (2 * a_au), rel=1e-10)  # E = −GM/2a
    lz = math.sqrt(gm.item() * a_au * (1 - PUBLISHED["e"] ** 2))
    assert ang.norm(dim=-1).mean().item() == pytest.approx(lz, rel=1e-10)  # |L| = √(GM a (1 − e²))
    speeds = v.norm(dim=-1)
    assert 7000 < speeds.max().item() * equation.AU_PER_YR_KMS < 8000  # S2 at pericentre: ≈ 7700 km/s


def test_precession_formula():
    # Mercury: a = 0.387098 AU, e = 0.205630, P = 0.2408467 yr → 42.98″ per century (Einstein 1915)
    mercury = equation.precession_per_orbit(torch.tensor(1.0), torch.tensor(0.387098), torch.tensor(0.205630))
    per_century = math.degrees(mercury.item()) * 3600 * 100 / 0.2408467
    assert per_century == pytest.approx(42.98, abs=0.05)
    # S2: 6πGM/(c² a (1 − e²)) ≈ 12′ per orbit (GRAVITY 2020)
    a_au = PUBLISHED["a"] / 1000 * PUBLISHED["R0"] * 1000
    s2 = equation.precession_per_orbit(torch.tensor(4.35e6), torch.tensor(a_au), torch.tensor(PUBLISHED["e"]))
    assert 11.0 < math.degrees(s2.item()) * 60 < 13.5


def test_autograd_jacobian_matches_finite_differences():
    data = tensors.load_s2()
    theta = tensors.unconstrained(elements())  # (1, 8)
    jac = solve.jacobian(theta, data)  # (1, R, 8)
    eps = 1e-6
    for k in range(8):
        step = torch.zeros_like(theta)
        step[0, k] = eps
        fd = (solve.residuals(theta + step, data) - solve.residuals(theta - step, data)) / (2 * eps)
        assert torch.allclose(jac[0, :, k], fd[0], rtol=1e-4, atol=1e-5)


def test_published_elements_fit_the_data():
    data = tensors.load_s2()
    chi2 = solve.residuals(tensors.unconstrained(elements()), data).pow(2).sum().item()
    # the published orbit goes through the points (χ²/N ≈ 4 without the reference-frame terms of the paper)
    assert chi2 / data["n_res"] < 5.0


@pytest.mark.slow
def test_batched_fit_recovers_synthetic_elements():
    gen = torch.Generator().manual_seed(3)
    data = tensors.synthetic_s2(tensors.load_s2(), elements()[0], gen)
    fit = solve.fit(data, n_starts=64, n_iter=40, gen=gen)
    best, err = fit["best"], fit["errors"]
    true = {**PUBLISHED, "P": equation.period(elements()).item()}
    for key in ("a", "e", "inc", "Omega", "omega", "tp", "M", "R0", "P"):
        assert abs(best[key] - true[key]) < 3.5 * err[key], key  # within the fit's own error bars
    assert err["M"] < 0.3 and err["R0"] < 0.25  # S2 alone weighs the black hole to a few per cent
    assert fit["chi2"].min().item() / data["n_res"] < 1.3  # pure Gaussian errors ⇒ χ²/N ≈ 1


@pytest.mark.slow
def test_real_s2_fit():
    fit = solve.fit(tensors.load_s2(), n_starts=64, n_iter=40, gen=torch.Generator().manual_seed(0))
    best = fit["best"]
    assert 3.9 < best["M"] < 4.7  # published 4.1–4.4 × 10⁶ M☉
    assert 7.8 < best["R0"] < 8.8  # published 8.2–8.35 kpc
    assert 15.8 < best["P"] < 16.2
    assert 0.87 < best["e"] < 0.90
