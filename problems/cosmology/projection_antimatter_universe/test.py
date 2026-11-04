"""Checks, written first (TDD): the criteria an antimatter universe must meet, as physical and mathematical laws.

Run with `pytest problems/cosmology/projection_antimatter_universe/test.py`. The hypothesis (P3: all particles are one
worldline, balanced globally with antimatter far away or across the Big Bang) is speculative; these tests check the
physics used to test it. Annihilation at domain walls scales with wall area per volume (1/d), a random sign sequence
has net-charge variance equal to its length, a closed worldline is neutral, the neutrino-mass floors follow from the
oscillation splittings, and bounds are applied the right way round.
"""

import math

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --- criterion 2: domains ---


def test_wall_area_per_volume_of_a_checkerboard_is_n_over_d():
    d = torch.tensor([10.0, 20.0, 50.0, 400.0], dtype=torch.float64)
    for n in (1, 2, 3, 5):
        av = equation.wall_area_per_volume(d, n)
        assert torch.allclose(av * d, torch.full_like(d, float(n)))  # n faces of area d^(n-1) per cell of volume d^n


def test_measured_wall_area_of_random_domains_falls_as_one_over_d():
    noise = torch.randn(512, 512, generator=gen(1), dtype=torch.float64)
    widths = torch.tensor([6.0, 12.0, 24.0], dtype=torch.float64)  # domain size d in pixels
    sign = equation.domains(noise, widths)  # (W, N, N) +1 matter, -1 antimatter
    av = equation.measured_wall_per_area(sign)  # (W,) wall length per unit area, pixels^-1
    prod = av * widths
    assert (prod.max() / prod.min()) < 1.1  # (wall per area) x d is constant: ~ 1/d


def test_annihilation_gamma_flux_scales_as_one_over_domain_size():
    c = tensors.cited()
    E = torch.logspace(-0.5, 5.5, 40, dtype=torch.float64)  # MeV
    d = torch.tensor([100.0, 200.0, 1000.0, 4000.0], dtype=torch.float64)  # Mpc
    J = equation.annihilation_intensity(E, d, c, speed_factor=1.0, z_max=1000.0)  # (D, E)
    on = J[0] > 0  # energies the redshifted pion photons reach
    assert on.sum() > 10
    assert torch.allclose(
        J[0, on] / J[1, on], torch.full_like(E[on], 2.0), rtol=1e-12
    )  # twice the size, half the walls
    assert torch.allclose(J[0, on] / J[3, on], torch.full_like(E[on], 40.0), rtol=1e-12)


def test_photon_spectrum_of_one_annihilation_holds_n_gamma_photons_and_energy_below_2_mp():
    E = torch.logspace(0, 3.2, 6000, dtype=torch.float64)  # MeV
    S = equation.pion_decay_spectrum(E)  # photons per MeV per annihilation
    N = torch.trapezoid(S, E)
    assert abs(N - equation.N_GAMMA) < 0.02 * equation.N_GAMMA
    assert torch.trapezoid(S * E, E) < 2 * 938.272  # photons carry less than the rest energy of p + pbar


def test_redshifted_density_matches_a_closed_form_for_a_line_and_constant_rate():
    # a narrow line at E0 emitted at a constant comoving rate R in an Einstein-de Sitter universe, H = H0 (1+z)^(3/2):
    # n(E) = int dz R delta(E(1+z) - E0) / H(z) = R / (H0 E0) (E / E0)^(1/2) for E < E0
    E0, R, H0, w = 100.0, 1.0, 1.0, 0.1
    E = torch.tensor([10.0, 25.0, 50.0], dtype=torch.float64)
    z, dz = equation.z_grid(20.0, 40000)
    line = lambda e: torch.exp(-((e - E0) ** 2) / (2 * w**2)) / (w * math.sqrt(2 * math.pi))  # noqa: E731
    n = equation.redshifted_density(E, z, dz, torch.full_like(z, R), H0 * (1 + z) ** 1.5, line)
    assert torch.allclose(n, R / (H0 * E0) * (E / E0) ** 0.5, rtol=2e-3)


def test_minimum_domain_size_is_near_the_horizon_scale():
    c = tensors.cited()
    res = equation.minimum_domain_size(c, speed_factor=1.0, z_max=1000.0)
    assert 3000.0 < res["d_min_mpc"] < 1e5  # Gpc scale: comparable to the 14.3 Gpc horizon radius
    faster = equation.minimum_domain_size(c, speed_factor=3.0, z_max=1000.0)
    assert faster["d_min_mpc"] == pytest.approx(3.0 * res["d_min_mpc"], rel=1e-9)  # flux ~ wall speed


# --- criterion 1: nothing mixed in nearby ---


def test_cluster_limit_on_the_antimatter_fraction():
    c = tensors.cited()
    f_max = equation.cluster_antimatter_limit(c)
    assert 1e-9 < f_max < 1e-5  # Coma: at most about one antiproton in 10^7 protons
    flux = equation.cluster_flux(torch.tensor([f_max, 2 * f_max], dtype=torch.float64), c)
    assert flux[0] == pytest.approx(c["coma_flux_ul"], rel=1e-9) and flux[1] > c["coma_flux_ul"]


def test_beta_model_integral_of_n_squared_matches_closed_form_for_beta_one():
    # beta = 1: int 4 pi r^2 n0^2 / (1 + r^2/rc^2)^3 dr from 0 to infinity = pi^2 n0^2 rc^3 / 4
    n0, rc = 2.0, 3.0
    val = equation.emission_measure(n0, rc, 1.0, r_max=2000 * rc)
    assert val == pytest.approx(math.pi**2 * n0**2 * rc**3 / 4, rel=1e-3)


# --- criterion 3: fine-tuning ---


def test_random_walk_net_charge_variance_equals_the_number_of_crossings():
    M = torch.tensor([10, 100, 1000])
    Q = equation.random_net_charge(M, 20000, gen(2))  # (M, B) sums of M random +-1
    var = Q.double().var(1)
    assert torch.allclose(var, M.double(), rtol=0.04)
    assert ((Q - M[:, None]) % 2 == 0).all()  # parity: Q = M mod 2


def test_balanced_loop_patch_variance_is_the_bridge_formula():
    T, M = 200, torch.tensor([20, 100, 180])
    Q = equation.patch_net_charge_balanced(T, M, 20000, gen(3))  # patch of M crossings out of a loop with Q_total = 0
    var = Q.double().var(1)
    assert torch.allclose(var, equation.bridge_variance(M, T).double(), rtol=0.05)  # M (T - M) / (T - 1)
    assert torch.equal(equation.patch_net_charge_balanced(T, torch.tensor([T]), 5, gen(4)), torch.zeros(1, 5).long())


def test_large_deviation_log_probability():
    M = torch.tensor([400.0, 1000.0], dtype=torch.float64)
    Q = torch.tensor([40.0, 300.0], dtype=torch.float64)
    exact = equation.log10_tail_exact(Q, M)  # binomial tail with lgamma
    approx = equation.log10_tail(Q, M)  # Cramer rate function
    assert (approx >= exact - 0.01).all() and (approx - exact < 0.25 * approx.abs() + 1.5).all()
    big = equation.log10_tail(torch.tensor([1e70], dtype=torch.float64), torch.tensor([1e89], dtype=torch.float64))
    assert big.item() == pytest.approx(-1e140 / (2e89 * math.log(10)), rel=1e-6)  # Gaussian tail -Q^2 / 2M
    allsame = equation.log10_tail(torch.tensor([9e79], dtype=torch.float64), torch.tensor([9e79], dtype=torch.float64))
    assert allsame.item() == pytest.approx(-9e79 * math.log10(2), rel=1e-9)  # every crossing the same sign: 2^-M


# --- criterion 4: CPT mirror ---


def test_balanced_closed_loop_has_total_charge_zero():
    t = tensors.closed_loops(64, 800, modes=8, dims=5, generator=gen(5))[..., 0]  # (B, S) global time of closed loops
    levels = torch.linspace(-0.9, 0.9, 19, dtype=torch.float64)
    up, down = equation.crossings(t, levels, closed=True)
    assert torch.equal(up - down, torch.zeros_like(up))  # neutral on every slice


def test_winding_loop_reads_plus_w_here_and_minus_w_in_the_mirror():
    w = torch.tensor([0, 1, 3, 7])
    eta = tensors.winding_conformal_loop(w, 1600, wiggle=0.25, generator=gen(6))  # unwrapped conformal time
    ours = torch.linspace(0.1, 0.9, 9, dtype=torch.float64)
    q_ours, q_mirror = equation.charges_across_mirror(eta, ours, period=2.0)  # (W, L) each with its own arrow of time
    assert torch.equal(q_ours, w[:, None].expand(-1, 9))
    assert torch.equal(q_mirror, -w[:, None].expand(-1, 9))
    assert torch.equal(q_ours + q_mirror, torch.zeros_like(q_ours))  # globally balanced


def test_neutrino_mass_sum_minima_follow_from_the_splittings():
    c = tensors.cited()
    m0 = torch.zeros(1, dtype=torch.float64)
    s_no = equation.sum_mnu(m0, "normal", c)
    s_io = equation.sum_mnu(m0, "inverted", c)
    no_ref = math.sqrt(c["dm21"]) + math.sqrt(c["dm31_no"])
    io_ref = math.sqrt(-c["dm32_io"]) + math.sqrt(-c["dm32_io"] - c["dm21"])
    assert s_no.item() == pytest.approx(no_ref, rel=1e-12) and 0.058 < s_no.item() < 0.060
    assert s_io.item() == pytest.approx(io_ref, rel=1e-12) and 0.098 < s_io.item() < 0.101
    m = torch.logspace(-4, 0, 50, dtype=torch.float64)  # the sum grows with the lightest mass and tends to 3 m
    for order in ("normal", "inverted"):
        s = equation.sum_mnu(m, order, c)
        assert (s.diff() > 0).all() and abs(s[-1] / (3 * m[-1]) - 1) < 0.01


def test_bounds_are_applied_correctly_to_toy_points():
    c = tensors.cited()
    m = torch.tensor([0.0, 0.0, 0.3, 0.6], dtype=torch.float64)
    order = ["normal", "inverted", "normal", "normal"]
    s = torch.stack([equation.sum_mnu(m[i : i + 1], order[i], c)[0] for i in range(4)])
    mb = torch.stack([equation.m_beta(m[i : i + 1], order[i], c)[0] for i in range(4)])
    assert equation.allowed(s, c["desi_dr2_sum_mnu"]).tolist() == [True, False, False, False]
    assert equation.allowed(s, c["planck_sum_mnu"]).tolist() == [True, True, False, False]
    assert equation.allowed(mb, c["katrin_2025"]).tolist() == [True, True, True, False]  # 0.6 eV is above 0.45
    assert equation.allowed(mb, c["katrin_2022"]).tolist() == [True, True, True, True]
    r = torch.tensor([0.0, 0.01, 0.05], dtype=torch.float64)
    assert equation.allowed(r, c["bicep_keck_r"]).tolist() == [True, True, False]
    assert equation.allowed(torch.tensor([0.036]), 0.036).item() is False  # a limit is strict


@pytest.mark.slow
def test_run_accepts_dimensions_as_parameters(tmp_path):
    from . import main

    params = {"dims": 6, "speed_factor": 1.0, "z_max": 600.0, "winding": 2}
    out = main.run(params, tmp_path, frames=6)
    assert out.exists() and out.stat().st_size > 0
