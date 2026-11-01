"""Physics checks for three research-inspired problems: QEC threshold, PTA Hellings–Downs, ICF ignition."""

import itertools
import math

import pytest
import torch

from problems.astro.pulsar_timing_hellings_downs import equation as pta_eq
from problems.astro.pulsar_timing_hellings_downs import solve as pta_solve
from problems.astro.pulsar_timing_hellings_downs import tensors as pta_t
from problems.nuclear.fusion_ignition import equation as icf_eq
from problems.nuclear.fusion_ignition import solve as icf_solve
from problems.nuclear.fusion_ignition import tensors as icf_t
from problems.quantum.error_correction_threshold import equation as qec_eq
from problems.quantum.error_correction_threshold import solve as qec_solve
from problems.quantum.error_correction_threshold import tensors as qec_t

CPU = torch.device("cpu")


# --- quantum error correction: repetition code ------------------------------------------------------


def test_binomial_formula_small_cases():
    p = torch.tensor([0.01, 0.1, 0.3], dtype=torch.float64)
    assert torch.allclose(qec_eq.binomial_logical_error(p, 3), 3 * p**2 - 2 * p**3)
    assert torch.allclose(qec_eq.binomial_logical_error(torch.tensor(0.5), 7), torch.tensor(0.5, dtype=torch.float64))


@pytest.mark.parametrize("d", [3, 5, 7])
@pytest.mark.parametrize("p", [0.1, 0.25])
def test_majority_vote_matches_exact_binomial(d, p):
    gen = torch.Generator().manual_seed(d)
    shots = 200_000
    p_l, _ = qec_solve.run_point(d, p, 0.0, 1, shots, gen, CPU)
    exact = qec_eq.binomial_logical_error(torch.tensor(p), d).item()
    sigma = math.sqrt(exact * (1 - exact) / shots)
    assert abs(p_l - exact) < 4 * sigma + 1e-12


def test_space_time_decoder_reduces_to_majority_vote_without_misreads():
    """One round, checks that never lie (huge misread weight): min-weight decoding = majority vote."""
    d, n = 7, 5000
    gen = torch.Generator().manual_seed(1)
    e, _ = qec_t.sample_noise(n, 1, d, 0.3, 0.0, gen, CPU)
    x = qec_eq.accumulate(e)
    sigma = qec_eq.syndrome(x, qec_eq.parity_check(d))
    fails = qec_solve.decode_fails(x, sigma, qec_t.mismatch_table(d).float(), 1.0, 100.0)
    assert torch.equal(fails.bool(), qec_eq.majority_vote_fails(x[:, -1]))


def test_space_time_decoder_is_exact_minimum_weight():
    """Compare the dynamic programme with brute force over every data-error history (d = 3, R = 3)."""
    d, rounds, w_d, w_m = 3, 3, 1.0, 1.3
    h = qec_eq.parity_check(d)
    gen = torch.Generator().manual_seed(2)
    e, m = qec_t.sample_noise(300, rounds, d, 0.2, 0.2, gen, CPU)
    x = qec_eq.accumulate(e)
    sigma = qec_eq.syndrome(x, h, m)
    fails = qec_solve.decode_fails(x, sigma, qec_t.mismatch_table(d).float(), w_d, w_m)

    histories = torch.tensor(list(itertools.product([0, 1], repeat=rounds * d))).view(-1, rounds, d)  # (2^9, R, d)
    xh = qec_eq.accumulate(histories.bool())  # candidate accumulated errors
    sh = qec_eq.syndrome(xh, h)  # their noiseless syndromes
    for s in range(x.shape[0]):
        cost = w_d * histories.sum((1, 2)) + w_m * (sh != sigma[s]).sum((1, 2))  # (2^9,)
        final = xh[:, -1]
        best_true = cost[(final == x[s, -1]).all(1)].min()
        best_flip = cost[(final == 1 - x[s, -1]).all(1)].min()
        expected = 1.0 if best_flip < best_true else (0.5 if best_flip == best_true else 0.0)
        assert fails[s].item() == pytest.approx(expected)


def test_curves_cross_near_threshold_and_lambda_above_one_below_it():
    ds = [3, 5, 7]
    ps = torch.tensor([0.04, 0.07, 0.09, 0.11, 0.13, 0.16], dtype=torch.float64)
    res = qec_solve.sweep(ds, ps, 1.0, "phenomenological", 6000, seed=3, device=CPU)
    pl = res["p_logical"]
    for a in range(len(ds) - 1):
        pc = qec_solve.crossing(ps, pl[a], pl[a + 1])
        assert 0.08 < pc < 0.14, f"d = {ds[a]}/{ds[a + 1]} crossing at {pc}"
    lam = qec_solve.suppression(res["eps"], pl, res["shots"])
    assert torch.all(lam[:, 0] > 1.5)  # p = 0.04, well below threshold: strong suppression
    assert torch.all(pl[:, -1][1:] > pl[:, -1][:-1])  # p = 0.16, above threshold: bigger codes are worse


# --- pulsar timing array: Hellings–Downs ----------------------------------------------------------------


def test_hellings_downs_limits_and_minimum():
    c = torch.cos(torch.deg2rad(torch.tensor([0.0, 180.0], dtype=torch.float64)))
    assert torch.allclose(pta_eq.hellings_downs(c), torch.tensor([0.5, 0.25], dtype=torch.float64))
    zeta = torch.linspace(0.0, 180.0, 180001, dtype=torch.float64)
    gamma = pta_eq.hellings_downs(torch.cos(torch.deg2rad(zeta)))
    z_min, g_min = pta_eq.hd_minimum()
    assert zeta[gamma.argmin()].item() == pytest.approx(z_min, abs=0.01)
    assert gamma.min().item() == pytest.approx(g_min, abs=1e-9)
    assert z_min == pytest.approx(82.48, abs=0.01)
    assert g_min == pytest.approx(0.5 - 1.5 * math.exp(-5 / 6))


def test_hellings_downs_equals_sky_integral_of_antenna_patterns():
    gen = torch.Generator().manual_seed(4)
    zeta = torch.tensor([5.0, 40.0, 82.5, 130.0, 175.0], dtype=torch.float64)
    numeric = pta_solve.hd_from_sky_integral(zeta, 400_000, gen)
    exact = pta_eq.hellings_downs(torch.cos(torch.deg2rad(zeta)))
    assert torch.allclose(numeric, exact, atol=0.006)


def test_polarisation_tensors_are_transverse_traceless_and_orthogonal():
    gen = torch.Generator().manual_seed(5)
    omega = pta_t.random_directions(100, gen)
    e = pta_eq.polarisation_tensors(omega, torch.rand(100, generator=gen, dtype=torch.float64) * math.pi)
    assert torch.allclose(torch.einsum("mAii->mA", e), torch.zeros(100, 2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(
        torch.einsum("mAij,mj->mAi", e, omega), torch.zeros(100, 2, 3, dtype=torch.float64), atol=1e-12
    )
    gram = torch.einsum("mAij,mBij->mAB", e, e)
    assert torch.allclose(gram, 2 * torch.eye(2, dtype=torch.float64).expand(100, 2, 2), atol=1e-12)


def test_antenna_patterns_sky_average_and_exact_power():
    gen = torch.Generator().manual_seed(6)
    pulsar = pta_t.random_directions(1, gen)
    omega = pta_t.random_directions(200_000, gen)
    psi = torch.rand(200_000, generator=gen, dtype=torch.float64) * math.pi
    f = pta_eq.antenna_patterns(pulsar, omega, pta_eq.polarisation_tensors(omega, psi))  # (M, 2, 1)
    power = f.pow(2).sum(1)[:, 0]
    cos_theta = omega @ pulsar[0]
    assert torch.allclose(power, 0.25 * (1 - cos_theta) ** 2, atol=1e-9)  # Σ_A F² = ¼(1 − Ω̂·p̂)², any ψ
    assert power.mean().item() == pytest.approx(1 / 3, abs=0.004)


def test_simulated_correlations_prefer_hellings_downs():
    gen = torch.Generator().manual_seed(7)
    n = 50
    pulsars = pta_t.random_directions(n, gen)
    r = pta_solve.simulate(pulsars, pta_t.pulsar_distances(n, gen), pta_t.time_grid(96, 15.0), 12, 500, 6, 0.3, gen)
    zeta, rho = pta_solve.pair_correlations(r, pta_eq.separation_cos(pulsars))
    edges = pta_t.bin_edges(10)
    binned = pta_solve.bin_by_angle(zeta, rho, edges)
    fits = pta_solve.fit_models((edges[1:] + edges[:-1]) / 2, binned.mean(0), pta_solve.bin_errors(binned))
    chi_hd, chi_mono, chi_dip = (fits[k][1] for k in ("Hellings–Downs", "monopole", "dipole"))
    assert chi_hd < chi_mono / 5 and chi_hd < chi_dip / 5
    assert 0.5 < fits["Hellings–Downs"][0] < 1.2  # amplitude ≈ 1, diluted by white noise


# --- inertial-confinement fusion: hot-spot ignition -----------------------------------------------------


def test_dt_reactivity_values_and_peak():
    T = torch.tensor([10.0, 20.0], dtype=torch.float64)
    sv = icf_eq.sigma_v(T)
    assert sv[0].item() == pytest.approx(1.1e-22, rel=0.05)  # ≈ 1.1e-22 m³/s near 10 keV
    assert sv[1].item() == pytest.approx(4.3e-22, rel=0.05)
    grid = torch.linspace(5.0, 100.0, 9501, dtype=torch.float64)
    s = icf_eq.sigma_v(grid)
    assert 50.0 < grid[s.argmax()].item() < 80.0
    assert 7e-22 < s.max().item() < 1e-21  # peak of order 1e-21 m³/s


def test_ideal_ignition_temperature():
    assert 4.0 < icf_eq.ideal_ignition_temperature() < 4.7


def test_balance_boundary_moves_out_when_losses_grow():
    T = torch.linspace(6.0, 30.0, 25, dtype=torch.float64)
    base = icf_solve.balance_boundary(T, 100.0, 1.0)
    lossy = icf_solve.balance_boundary(T, 100.0, 1.5)
    assert torch.isfinite(base).all() and torch.isfinite(lossy).all()
    assert torch.all(lossy > base)
    # at the boundary heating balances losses
    margin = icf_eq.heating_balance(T, base, 100.0, 1.0)
    scale = icf_eq.powers(T, base, torch.zeros_like(T), 100.0)["alpha"]
    assert torch.all(margin.abs() / scale < 1e-6)


def test_integrated_ignition_region_shrinks_with_more_losses():
    t0, r0 = icf_t.initial_grid(8, 10, (4.0, 16.0), (0.05, 1.5))
    y0 = icf_t.initial_state(t0, r0)
    results = {}
    for loss in (1.0, 1.6):
        _, states = icf_solve.integrate(y0, 100.0, loss, 2.0, 120.0, 0.25, 20)
        assert torch.isfinite(states).all()
        ign = icf_solve.ignited(states)
        results[loss] = (ign, icf_solve.min_ignition_rho_r(t0, r0, ign))
    ign1, first1 = results[1.0]
    ign2, first2 = results[1.6]
    assert ign1.sum() > ign2.sum() > 0
    assert torch.all(ign2 <= ign1)  # anything that ignites with more losses also ignites with fewer
    both = torch.isfinite(first1) & torch.isfinite(first2)
    assert both.any() and torch.all(first2[both] >= first1[both])
    assert not ign1.view(8, 10)[0].any()  # 4 keV starts never ignite: below the ignition temperature range
