import math

import torch

from problems.nuclear.binding_energy import equation as be_eq
from problems.nuclear.binding_energy import solve as be_solve
from problems.nuclear.binding_energy import tensors as be_t
from problems.nuclear.decay_chain import equation as dc_eq
from problems.nuclear.decay_chain import solve as dc_solve
from problems.nuclear.decay_chain import tensors as dc_t
from problems.nuclear.em_shower import equation as sh_eq
from problems.nuclear.em_shower import solve as sh_solve
from problems.nuclear.em_shower import tensors as sh_t
from problems.nuclear.four_momentum import equation as fm_eq
from problems.nuclear.four_momentum import solve as fm_solve
from problems.nuclear.four_momentum import tensors as fm_t
from problems.nuclear.neutrino_oscillations import equation as nu_eq
from problems.nuclear.neutrino_oscillations import solve as nu_solve
from problems.nuclear.neutrino_oscillations import tensors as nu_t
from problems.nuclear.radioactive_decay import equation as rd_eq
from problems.nuclear.radioactive_decay import solve as rd_solve
from problems.nuclear.reactor_criticality import equation as rc_eq
from problems.nuclear.reactor_criticality import solve as rc_solve
from problems.nuclear.reactor_criticality import tensors as rc_t
from problems.nuclear.rutherford_scattering import equation as ru_eq
from problems.nuclear.rutherford_scattering import solve as ru_solve
from problems.nuclear.rutherford_scattering import tensors as ru_t
from problems.nuclear.shell_model import solve as sm_solve
from problems.nuclear.shell_model import tensors as sm_t

CPU = torch.device("cpu")


# --- radioactive decay --------------------------------------------------------------------------


def test_decay_half_life_halves_the_exact_count():
    lam = rd_eq.decay_constant(3.0)
    assert math.isclose(math.log(2) / lam, 3.0)
    t = torch.tensor([0.0, 3.0, 6.0], dtype=torch.float64)
    assert torch.allclose(rd_eq.exact_count(1000, lam, t), torch.tensor([1000.0, 500.0, 250.0], dtype=torch.float64))


def test_decay_monte_carlo_mean_matches_exponential():
    lam, dt, n_steps = rd_eq.decay_constant(2.0), 0.1, 60
    alive0 = torch.ones(20, 30, 30, dtype=torch.bool)  # 20 repeated experiments of a 30 × 30 grid
    history = rd_solve.simulate(alive0, lam, dt, n_steps, seed=3)  # (T, 20, 30, 30)
    counts = rd_eq.count(history).double()  # (T, 20)
    t = torch.arange(n_steps + 1, dtype=torch.float64) * dt
    n0 = 900
    survive = torch.exp(-lam * t)
    sigma_mean = torch.sqrt(n0 * survive * (1 - survive) / counts.shape[1])  # binomial σ of the mean of 20 runs
    deviation = (counts.mean(dim=1) - n0 * survive).abs()
    assert torch.all(deviation <= 4 * sigma_mean + 1e-9)
    assert torch.all(history[1:] <= history[:-1])  # decayed nuclei never come back


# --- decay chain --------------------------------------------------------------------------------


def chain(t1: float = 10.0, t2: float = 1.0):
    lam = dc_t.decay_constants([t1, t2], CPU)
    return lam, dc_eq.bateman_matrix(lam), dc_t.initial_populations(1000.0, 3, CPU)


def test_chain_matches_two_member_bateman_solution():
    lam, a, n0 = chain()
    t = dc_t.time_axis(50.0, 101, CPU)
    n = dc_solve.exact(a, n0, t)
    l1, l2 = lam[0], lam[1]
    n1 = 1000.0 * torch.exp(-l1 * t)
    n2 = 1000.0 * l1 / (l2 - l1) * (torch.exp(-l1 * t) - torch.exp(-l2 * t))
    assert torch.allclose(n[:, 0], n1, rtol=1e-10)
    assert torch.allclose(n[:, 1], n2, rtol=1e-9, atol=1e-9)


def test_chain_conserves_nuclei_and_rk4_agrees():
    _, a, n0 = chain()
    t_rk, n_rk = dc_solve.rk4(a, n0, 50.0, 500)
    n = dc_solve.exact(a, n0, t_rk)
    assert torch.allclose(n.sum(dim=1), torch.full_like(t_rk, 1000.0), rtol=1e-12)
    assert torch.allclose(n_rk, n, atol=1e-6)


def test_chain_transient_and_secular_equilibrium():
    lam, a, n0 = chain(10.0, 1.0)
    t = torch.tensor([150.0], dtype=torch.float64)
    act = dc_eq.activity(dc_solve.exact(a, n0, t), lam)[0]
    expected = lam[1] / (lam[1] - lam[0])
    assert math.isclose(act[1] / act[0], expected, rel_tol=1e-6)
    lam, a, n0 = chain(10000.0, 1.0)  # very long-lived parent → secular: a2 = a1
    act = dc_eq.activity(dc_solve.exact(a, n0, torch.tensor([30.0], dtype=torch.float64)), lam)[0]
    assert math.isclose(act[1] / act[0], 1.0, rel_tol=2e-4)


# --- binding energy -----------------------------------------------------------------------------

COEFF = {"a_V": 15.8, "a_S": 18.3, "a_C": 0.714, "a_A": 23.2, "a_P": 12.0}


def binding_table():
    z, n = be_t.grid(110, 176, CPU)
    return z, n, be_eq.binding_energy(z, n, COEFF)


def test_binding_per_nucleon_peaks_near_iron():
    z, n, b = binding_table()
    mass, _, best = be_solve.most_stable(b, z, n)
    a_peak, ba_peak = be_solve.peak(mass, best)
    assert 56 <= a_peak <= 62
    assert 8.6 < ba_peak < 8.9


def test_binding_valley_of_stability():
    z, n, b = binding_table()
    mass, z_best, _ = be_solve.most_stable(b, z, n)
    odd = (mass % 2 == 1) & (mass > 20) & (mass < 240)  # odd A: no pairing term to make Z jump
    formula = mass / (2 + 0.0154 * mass ** (2 / 3))
    assert torch.all((z_best[odd] - formula[odd]).abs() <= 1.0)


def test_binding_energies_of_known_nuclei():
    _, _, b = binding_table()
    for name, zz, a, ba in be_t.MEASURED:
        if a >= 40:  # the liquid drop is good to ~1 % for medium and heavy nuclei
            assert math.isclose(b[zz - 1, a - zz].item(), ba * a, rel_tol=0.01), name
    he4 = b[1, 2].item()
    assert 0.6 * 28.3 < he4 < 28.3  # ⁴He is famously under-bound by the liquid drop


# --- Rutherford scattering ----------------------------------------------------------------------


def rutherford_run(b: torch.Tensor, energy: float = 5.0):
    k = ru_t.coupling(79)
    d = k / energy
    m = ru_t.ALPHA_MASS
    start = 30 * d
    y0 = ru_t.initial_state(b, energy, k, start, CPU)
    dt = 2.2 * start / math.sqrt(2 * energy / m) / 4000
    paths, y_end, _ = ru_solve.simulate(y0, k, m, dt, 4000, len(b), r_detect=5 * d)
    return k, m, d, y0, paths, y_end


def test_rutherford_angle_vs_impact_parameter():
    energy = 5.0
    k = ru_t.coupling(79)
    b = torch.tensor([0.25, 0.5, 1.0, 2.0, 3.0], dtype=torch.float64) * k / energy
    _, _, _, _, _, y_end = rutherford_run(b, energy)
    theta = ru_eq.scattering_angle(y_end[:, 1])
    expected = ru_eq.rutherford_angle(b, energy, k)  # cot(θ/2) = 2Eb/k
    assert torch.allclose(theta, expected, atol=0.01)


def test_rutherford_conserves_energy_and_angular_momentum():
    k = ru_t.coupling(79)
    b = torch.tensor([0.0, 0.3, 1.0, 2.5], dtype=torch.float64) * k / 5.0
    _, m, _, y0, paths, _ = rutherford_run(b)
    e = ru_eq.energy(paths, k, m)  # (T, B)
    lz = ru_eq.angular_momentum(paths, m)
    assert torch.allclose(e, torch.full_like(e, 5.0), rtol=1e-6)
    assert torch.allclose(lz, lz[0].expand_as(lz), atol=1e-6 * m * float(b.max()))
    v_inf = math.sqrt(2 * 5.0 / m)
    assert torch.allclose(lz[0].abs(), m * v_inf * b, rtol=1e-12)


def test_rutherford_distance_of_closest_approach():
    k = ru_t.coupling(79)
    b = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64) * k / 5.0
    _, _, _, _, paths, _ = rutherford_run(b)
    r_min = paths[:, :, 0, :].norm(dim=-1).min(dim=0).values
    assert torch.allclose(r_min, ru_eq.closest_approach(b, 5.0, k), rtol=2e-3)


# --- neutrino oscillations ----------------------------------------------------------------------


def test_pmns_is_unitary_and_probabilities_sum_to_one():
    u = nu_eq.pmns(*(math.radians(x) for x in (33.4, 8.6, 49.0, 195.0)))
    eye = torch.eye(3, dtype=torch.complex128)
    assert torch.allclose(u @ u.conj().T, eye, atol=1e-14)
    m2 = nu_t.squared_masses(7.4e-5, 2.5e-3, CPU)
    l_over_e = torch.linspace(0.0, 40000.0, 500, dtype=torch.float64)
    p = nu_eq.probability(u, nu_eq.phases(m2, l_over_e))  # (K, β, α)
    assert torch.allclose(p.sum(dim=1), torch.ones(500, 3, dtype=torch.float64), atol=1e-12)  # Σ_β
    assert torch.allclose(p.sum(dim=2), torch.ones(500, 3, dtype=torch.float64), atol=1e-12)  # Σ_α
    assert torch.allclose(p[0], torch.eye(3, dtype=torch.float64), atol=1e-12)  # no change at L = 0


def test_neutrino_two_flavour_limit():
    theta23, dm31 = math.radians(40.0), 2.5e-3
    u = nu_eq.pmns(0.0, 0.0, theta23, 0.0)  # only ν_μ–ν_τ mixing
    m2 = nu_t.squared_masses(0.0, dm31, CPU)
    dist = torch.linspace(1.0, 5000.0, 400, dtype=torch.float64)
    p = nu_solve.oscillate(u, m2, dist, 1.0, alpha=1)
    assert torch.allclose(p[:, 2], nu_eq.two_flavour(theta23, dm31, dist), atol=1e-12)
    assert torch.allclose(p[:, 0], torch.zeros_like(dist), atol=1e-12)


def test_neutrino_cpt_relation():
    m2 = nu_t.squared_masses(7.4e-5, 2.5e-3, CPU)
    l_over_e = torch.linspace(0.0, 3000.0, 100, dtype=torch.float64)
    angles = [math.radians(x) for x in (33.4, 8.6, 49.0)]
    p_plus = nu_eq.probability(nu_eq.pmns(*angles, 1.2), nu_eq.phases(m2, l_over_e))
    p_minus = nu_eq.probability(nu_eq.pmns(*angles, -1.2), nu_eq.phases(m2, l_over_e))
    assert torch.allclose(p_plus, p_minus.transpose(1, 2), atol=1e-12)  # P(α→β; δ) = P(β→α; −δ)


# --- four-momentum ------------------------------------------------------------------------------


def decay(name: str, p_lab: float = 1.7):
    _, big_m, m1, m2, _, _ = fm_t.DECAYS[name]
    eta = fm_eq.metric(CPU)
    parent = fm_t.parent(big_m, p_lab, CPU)
    theta = fm_t.decay_angles(37, CPU)
    p_star = fm_eq.rest_frame_momentum(big_m, m1, m2)
    r1, r2 = fm_t.rest_frame_daughters(theta, p_star, big_m, m1, m2)
    l1, l2 = fm_solve.to_lab(parent, r1, r2)
    return big_m, m1, m2, eta, parent, r1, r2, l1, l2


def test_boost_preserves_metric_and_masses():
    for name in fm_t.DECAYS:
        big_m, m1, m2, eta, parent, _, _, l1, l2 = decay(name)
        lam = fm_eq.boost(parent[1:] / parent[0])
        assert torch.allclose(torch.einsum("am,ab,bn->mn", lam, eta, lam), eta, atol=1e-12)
        assert torch.allclose(fm_eq.minkowski(l1, l1, eta), torch.full((37,), m1**2, dtype=torch.float64), atol=1e-12)
        assert torch.allclose(fm_eq.minkowski(l2, l2, eta), torch.full((37,), m2**2, dtype=torch.float64), atol=1e-12)
        assert math.isclose(fm_eq.invariant_mass(parent, eta).item(), big_m, rel_tol=1e-12)


def test_invariant_mass_and_momentum_conservation():
    for name in fm_t.DECAYS:
        big_m, _, _, eta, parent, _, _, l1, l2 = decay(name)
        assert torch.allclose(fm_solve.reconstruct(l1, l2, eta), torch.full((37,), big_m, dtype=torch.float64))
        assert torch.allclose(l1 + l2, parent.expand_as(l1), atol=1e-12)


def test_rest_frame_momentum_formula():
    big_m, m1, m2, _, _, r1, r2, _, _ = decay("lambda")
    p_star = fm_eq.rest_frame_momentum(big_m, m1, m2)
    assert math.isclose(p_star, 0.101, abs_tol=1e-3)  # Λ → p π⁻: p* ≈ 101 MeV
    assert torch.allclose(r1[:, 0] + r2[:, 0], torch.full((37,), big_m, dtype=torch.float64))  # E1 + E2 = M
    assert torch.allclose(r1[:, 1:].norm(dim=1), torch.full((37,), p_star, dtype=torch.float64))
    assert math.isclose(fm_eq.rest_frame_momentum(0.134977, 0.0, 0.0), 0.134977 / 2)  # π⁰ → γγ: p* = M/2


# --- shell model --------------------------------------------------------------------------------


def shell_scheme(v_so: float, mass_number: int = 208, v0: float = 44.0):
    r, h = sm_t.radial_grid(300, 18.0, CPU)
    d2 = sm_t.second_difference(len(r), h, CPU)
    ell, j = sm_t.channels(8, CPU)
    radius = 1.27 * mass_number ** (1 / 3)
    energies = sm_solve.bound_levels(r, d2, ell, j, v0, v_so, radius, 0.67, 1.27)
    return sm_solve.level_scheme(energies, ell, j), ell, j


def test_shell_without_spin_orbit_gives_oscillator_magic_numbers():
    scheme, ell, j = shell_scheme(0.0)
    _, magic = sm_solve.largest_gaps(scheme, 5)
    assert {2, 8, 20, 40} <= set(int(x) for x in magic.tolist())
    # without spin–orbit the j = l ± ½ partners are degenerate
    e = sm_solve.bound_levels(*_shell_inputs(), 44.0, 0.0, 1.27 * 208 ** (1 / 3), 0.67, 1.27)
    assert torch.allclose(e[1], e[2], equal_nan=True)  # p3/2 and p1/2


def _shell_inputs():
    r, h = sm_t.radial_grid(300, 18.0, CPU)
    ell, j = sm_t.channels(8, CPU)
    return r, sm_t.second_difference(len(r), h, CPU), ell, j


def test_shell_spin_orbit_gives_real_magic_numbers():
    scheme, ell, j = shell_scheme(22.0)
    _, magic = sm_solve.largest_gaps(scheme, 7)
    assert set(int(x) for x in magic.tolist()) == {2, 8, 20, 28, 50, 82, 126}


def test_shell_levels_sorted_with_2j_plus_1_degeneracy():
    scheme, ell, j = shell_scheme(22.0)
    assert torch.all(torch.diff(scheme["energy"]) >= 0)
    occupancy = torch.diff(scheme["filled"], prepend=torch.zeros(1, dtype=scheme["filled"].dtype))
    assert torch.equal(occupancy, 2 * j[scheme["channel"]] + 1)
    first = scheme["channel"][0]
    assert ell[first] == 0 and j[first] == 0.5  # 1s1/2 is the lowest level
    k = {(int(a), float(b)): i for i, (a, b) in enumerate(zip(ell.tolist(), j.tolist()))}
    e = dict(zip(zip(scheme["channel"].tolist(), scheme["radial"].tolist()), scheme["energy"].tolist()))
    assert e[(k[(4, 4.5)], 0)] < e[(k[(4, 3.5)], 0)]  # 1g9/2 below 1g7/2: spin–orbit favours j = l + ½


# --- reactor criticality ------------------------------------------------------------------------

D, SA, NSF = 0.9, 0.066, 0.070


def slab(width: float, n: int = 200):
    x, h = rc_t.grid(width, n, CPU)
    m = rc_eq.loss_operator(rc_t.laplacian(n, h, CPU), D, SA)
    f = rc_eq.production_operator(n, NSF, CPU)
    return x, m, f


def test_reactor_k_matches_bare_slab_formula():
    for width in (30.0, 47.0, 60.0, 120.0):
        _, m, f = slab(width)
        assert math.isclose(rc_solve.k_eigen(m, f).item(), rc_eq.analytic_k(width, D, SA, NSF), rel_tol=1e-5)


def test_reactor_power_iteration_converges_to_cosine_mode():
    width = 60.0
    x, m, f = slab(width)
    ks, fluxes = rc_solve.power_iteration(m, f, rc_t.initial_guess(x, width), 200)
    assert math.isclose(ks[-1].item(), rc_solve.k_eigen(m, f).item(), rel_tol=1e-8)
    shape = torch.sin(math.pi * x / width)  # = cos(π x'/L) measured from the centre
    assert torch.allclose(fluxes[-1], shape / shape.max(), atol=1e-4)


def test_reactor_critical_width_and_batched_sizes():
    l_crit = rc_eq.critical_width(D, SA, NSF)
    _, m, f = slab(l_crit)
    assert math.isclose(rc_solve.k_eigen(m, f).item(), 1.0, rel_tol=1e-5)
    widths = torch.tensor([20.0, 40.0, 80.0], dtype=torch.float64)
    k = rc_solve.k_versus_width(widths, 150, D, SA, NSF)
    assert torch.allclose(k, rc_eq.analytic_k(widths, D, SA, NSF), rtol=1e-4)
    assert k[0] < 1 < k[2]


# --- electromagnetic shower ---------------------------------------------------------------------


def run_shower(e0: float, e_crit: float, stochastic: bool, seed: int = 0):
    t_max = sh_eq.heitler_t_max(e0, e_crit)
    depths = sh_t.depth_axis(t_max + 12.0, 20, CPU)
    result = sh_solve.shower(sh_t.primary(e0, CPU), depths, 20, e_crit, stochastic, seed=seed)
    return depths, result


def test_shower_heitler_maximum():
    for e0, e_crit in ((10000.0, 7.4), (1024.0 * 10.0, 10.0), (50000.0, 20.0)):
        depths, result = run_shower(e0, e_crit, stochastic=False)
        total = result["counts"].sum(dim=1)
        k = int(total.argmax())
        n_max, t_max = sh_eq.heitler_n_max(e0, e_crit), sh_eq.heitler_t_max(e0, e_crit)
        assert n_max / 2 < total[k].item() <= n_max
        assert abs(depths[k].item() - t_max) <= 1.0
    depths, result = run_shower(1024.0 * 10.0, 10.0, stochastic=False)  # E0/Ec a power of 2: exact
    total = result["counts"].sum(dim=1)
    assert total.max().item() == 1024 and depths[int(total.argmax())].item() == 10.0


def test_shower_energy_conservation():
    for stochastic in (False, True):
        _, result = run_shower(10000.0, 7.4, stochastic)
        balance = result["in_flight"] + result["deposited"]
        assert torch.allclose(balance, torch.full_like(balance, 10000.0), rtol=1e-12)
    _, result = run_shower(10000.0, 7.4, stochastic=False)
    assert result["counts"][-1].sum() == 0  # fixed lengths: everything has been absorbed by the end


def test_shower_stochastic_profile():
    depths, result = run_shower(10000.0, 7.4, stochastic=True, seed=1)
    total = result["counts"].sum(dim=1).double()
    t_mean = (total * depths).sum() / total.sum()  # centre of the longitudinal profile
    assert sh_eq.heitler_t_max(10000.0, 7.4) - 1 < t_mean < sh_eq.heitler_t_max(10000.0, 7.4) + 6
    assert total.max() < sh_eq.heitler_n_max(10000.0, 7.4)
