"""Checks, written first (TDD): what any correct telescope test of the projection hypothesis must satisfy.

Run with `pytest problems/cosmology/projection_telescope_tests/test.py`. The hypothesis is speculative; these tests
check the physics and the statistics used to confront it with data. If gravity spreads into D spacetime dimensions,
a gravitational wave's amplitude falls as d^-(D-2)/2, so D = 4 must give back d_GW = d_EM. The GW likelihood must be
a normalised density and the posterior of D must integrate to one; a fit to synthetic distances must find the
injected D. The Schwarzschild shadow radius is sqrt(27) GM/c^2 both from the photon sphere and from ray tracing; a
fold inside the photon sphere leaves it unchanged, and the bound on the fold radius is where the shadow growth meets
the allowed deviation.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def test_four_dimensions_give_back_the_electromagnetic_distance():
    d_em = torch.tensor([1.0, 10.0, 40.7, 1000.0], dtype=F64)
    four = torch.tensor(4.0, dtype=F64)
    assert torch.allclose(equation.gw_distance(d_em, four), d_em)  # unscreened
    # screened (D > 4 only): well inside R_c nothing leaks, and a large R_c switches leakage off at any distance
    small = equation.gw_distance(d_em * 1e-9, torch.tensor(5.0, dtype=F64), R_c=torch.tensor(20.0, dtype=F64), n=1.0)
    assert torch.allclose(small, d_em * 1e-9, rtol=1e-3)
    big = equation.gw_distance(d_em, torch.tensor(5.0, dtype=F64), R_c=torch.tensor(1e12, dtype=F64), n=1.0)
    assert torch.allclose(big, d_em, rtol=1e-4)


@pytest.mark.parametrize("D", [3.0, 4.0, 5.0, 6.5, 10.0])
def test_amplitude_falls_with_exponent_d_minus_two_over_two(D):
    d = torch.logspace(0, 3, 50, dtype=F64)  # 1 ... 1000 Mpc
    h = equation.strain_amplitude(d, torch.tensor(D, dtype=F64))
    slope = torch.diff(torch.log(h)) / torch.diff(torch.log(d))
    assert torch.allclose(slope, torch.full_like(slope, -(D - 2) / 2))
    # gauss's law in D - 1 spatial dimensions: energy flux ~ h^2 ~ 1 / area of a (D - 2)-sphere ~ d^-(D - 2)
    assert torch.allclose(2 * slope, torch.full_like(slope, -(D - 2.0)))


def test_screened_amplitude_has_the_right_limits():
    d = torch.logspace(-3, 6, 300, dtype=F64)
    D, Rc = torch.tensor(5.0, dtype=F64), torch.tensor(30.0, dtype=F64)
    dg = equation.gw_distance(d, D, R_c=Rc, n=2.0)
    slope = torch.diff(torch.log(dg)) / torch.diff(torch.log(d))
    assert abs(slope[:5].mean() - 1.0) < 1e-3  # d << R_c: ordinary 1/d
    assert abs(slope[-5:].mean() - (D - 2) / 2) < 1e-2  # d >> R_c: leaking, d^((D-2)/2)


def test_gw_likelihood_is_normalised():
    gw = tensors.load_gw_samples()
    assert gw["d"].shape[0] > 4000 and float(gw["d"].min()) > 0
    x = torch.linspace(0.1, 150.0, 6000, dtype=F64)
    L = equation.gw_likelihood(x, gw["d"], gw["w"], gw["h"])
    assert (L >= 0).all()
    assert abs(torch.trapezoid(L, x).item() - 1.0) < 2e-3
    # the d^2 volume prior is divided out: the likelihood's mean lies closer in than the posterior mean
    assert torch.trapezoid(x * L, x) < gw["d"].mean() - 1.0


def test_posterior_of_d_is_normalised_and_peaks_near_four():
    gw = tensors.load_gw_samples()
    em = tensors.em_distance("sbf")
    Ds = torch.linspace(3.0, 5.0, 401, dtype=F64)
    post = solve.posterior_D(Ds, gw, em)
    assert abs(torch.trapezoid(post["pdf"], Ds).item() - 1.0) < 1e-6
    lo, hi = post["interval"]
    assert lo < post["map"] < hi and lo < 4.0 < hi  # GR in 3 + 1 inside the 68% interval
    assert hi - lo < 0.4


def test_fit_recovers_an_injected_dimension():
    """Synthetic: d_EM known, a fake GW distance posterior centred on d_GW(D_true) with 5% error."""
    gen = torch.Generator().manual_seed(4)
    for D_true in (3.8, 4.0, 4.3):
        syn = tensors.synthetic_gw(40.0, torch.tensor(D_true, dtype=F64), 0.05, 6000, gen)
        em = {"d": 40.0, "sigma": 0.4, "label": "synthetic"}
        fit = solve.fit_D(syn, em, D0=4.6)  # autograd from a wrong start
        assert abs(fit["D"] - D_true) < 3 * fit["sigma"] + 0.01, (D_true, fit)
        assert fit["sigma"] < 0.05


def test_schwarzschild_photon_sphere_and_shadow():
    rs = 1.0
    b, r_ph = equation.critical_impact_parameter(torch.tensor([rs], dtype=F64))
    assert abs(r_ph.item() - 1.5) < 1e-4  # photon sphere at 3 GM/c^2
    assert abs(b.item() - math.sqrt(27) / 2) < 1e-6  # shadow radius sqrt(27) GM/c^2 = (sqrt(27)/2) r_s


@pytest.mark.slow
def test_ray_tracing_gives_the_same_shadow_edge():
    edge = solve.traced_shadow_edge(torch.tensor([0.0, 1.2, 2.0, 3.0], dtype=F64))  # fold radius / r_s (0 = none)
    exact = equation.fold_shadow_radius(torch.tensor([0.0, 1.2, 2.0, 3.0], dtype=F64))
    assert abs(exact[0].item() - math.sqrt(27) / 2) < 1e-9
    assert torch.allclose(edge, exact, rtol=2e-3), (edge, exact)


def test_fold_inside_photon_sphere_is_invisible_and_outside_grows():
    r = torch.linspace(1.0, 4.0, 301, dtype=F64)
    delta = equation.fold_delta(r)
    assert (delta[r <= 1.5].abs() < 1e-12).all()
    assert (torch.diff(delta[r > 1.5]) > 0).all()
    assert abs(delta[-1].item() - (4 / math.sqrt(0.75) / (math.sqrt(27) / 2) - 1)) < 1e-9


def test_kerr_shadow_band():
    a = torch.tensor([1e-4, 0.5, 0.9, 0.998], dtype=F64)
    inc = torch.tensor([17.0, 60.0, 90.0], dtype=F64)
    d = equation.kerr_delta(a, inc)  # (A, I) fractional change of the area-equivalent radius
    assert d[0].abs().max() < 1e-3  # a -> 0 is Schwarzschild, sqrt(27)
    assert (d <= 1e-4).all() and d.min() > -0.08  # spin only shrinks the shadow, by < 8%
    assert (torch.diff(d, dim=0) < 0).all()  # faster spin, smaller shadow
    # seen from the pole, an extremal hole's shadow is a circle of radius 2 (1 + sqrt 2) GM/c^2
    pole = equation.kerr_delta(torch.tensor([1 - 1e-7], dtype=F64), torch.tensor([1.0], dtype=F64), n=20000)
    assert abs(pole.item() - (2 * (1 + math.sqrt(2)) / math.sqrt(27) - 1)) < 2e-3


def test_bounds_are_applied_correctly():
    for allowed in (0.05, 0.1, 0.3):
        r_b = equation.fold_bound(torch.tensor(allowed, dtype=F64)).item()
        assert r_b > 1.5
        assert abs(equation.fold_delta(torch.tensor(r_b, dtype=F64)).item() - allowed) < 1e-8
        assert equation.fold_delta(torch.tensor(r_b * 0.99, dtype=F64)) < allowed
        assert equation.fold_delta(torch.tensor(r_b * 1.01, dtype=F64)) > allowed
    assert equation.fold_bound(torch.tensor(-0.02, dtype=F64)).item() == pytest.approx(1.5)  # only inside
    sh = solve.shadow_tests(tensors.load_eht(), spin_inc_max=90.0)
    for row in sh["rows"]:
        assert row["allowed"] == pytest.approx(row["delta_up"] - row["kerr_min"])  # spin may hide a fold
        assert row["r_bound"] >= 1.5


def test_supernova_leakage_model_reduces_to_lcdm_and_dgp():
    z = torch.linspace(0.0, 2.0, 50, dtype=F64)
    Om = torch.tensor(0.3, dtype=F64)
    E0 = equation.leaky_expansion(z, Om, torch.tensor(0.0, dtype=F64))
    assert torch.allclose(E0, (0.3 * (1 + z) ** 3 + 0.7).sqrt())  # alpha = 0: flat Lambda-CDM
    E1 = equation.leaky_expansion(z, Om, torch.tensor(1.0, dtype=F64))
    orc = (0.7 / 2) ** 2  # flat DGP: Omega_m + 2 sqrt(Omega_rc) = 1
    assert torch.allclose(E1, math.sqrt(orc) + (orc + 0.3 * (1 + z) ** 3).sqrt())
    assert torch.allclose(E1[0], torch.tensor(1.0, dtype=F64))
