import math

import pytest
import torch

from problems.astro.black_hole_geodesics import equation as bh_eq
from problems.astro.black_hole_geodesics import solve as bh_solve
from problems.astro.black_hole_geodesics import tensors as bh_t
from problems.astro.escape_velocity import equation as ev_eq
from problems.astro.escape_velocity import solve as ev_solve
from problems.astro.escape_velocity import tensors as ev_t
from problems.astro.gravitational_lensing import equation as gl_eq
from problems.astro.gravitational_lensing import solve as gl_solve
from problems.astro.gravitational_lensing import tensors as gl_t
from problems.astro.hawking_radiation import equation as hr_eq
from problems.astro.hawking_radiation import solve as hr_solve
from problems.astro.hubble_expansion import equation as hb_eq
from problems.astro.hubble_expansion import solve as hb_solve
from problems.astro.hubble_expansion import tensors as hb_t
from problems.astro.kepler_orbits import equation as kp_eq
from problems.astro.kepler_orbits import solve as kp_solve
from problems.astro.kepler_orbits import tensors as kp_t
from problems.astro.lagrange_points import equation as lp_eq
from problems.astro.lagrange_points import solve as lp_solve
from problems.astro.lagrange_points import tensors as lp_t
from problems.astro.minkowski_diagram import equation as mk_eq
from problems.astro.minkowski_diagram import solve as mk_solve
from problems.astro.minkowski_diagram import tensors as mk_t
from problems.astro.tidal_tensor import equation as td_eq
from problems.astro.tidal_tensor import solve as td_solve
from problems.astro.tidal_tensor import tensors as td_t

CPU = torch.device("cpu")


# --- Kepler's laws -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kepler_run():
    GM = kp_t.GM_SUN
    a, e = kp_t.orbit_shapes(4, 0.6)
    y0 = kp_t.initial_state(a.double(), e.double(), GM, CPU)
    times, traj = kp_solve.solve(y0, GM, t_max=3.5, dt=0.0005)
    return GM, a.double(), times, traj


def test_kepler_third_law(kepler_run):
    GM, a, times, traj = kepler_run
    T = kp_solve.measured_periods(times, traj)
    assert torch.isfinite(T).all()
    assert torch.allclose(T**2 / a**3, torch.full_like(a, 4 * math.pi**2 / GM), rtol=1e-3)


def test_kepler_areal_velocity_energy_and_angular_momentum_conserved(kepler_run):
    GM, _, _, traj = kepler_run
    for q in (kp_eq.areal_velocity(traj), kp_eq.energy(traj, GM), kp_eq.angular_momentum(traj)):
        assert torch.allclose(q, q[0].expand_as(q), rtol=1e-5)


def test_kepler_semi_major_axis_from_energy(kepler_run):
    GM, a, _, traj = kepler_run
    assert torch.allclose(kp_eq.semi_major_axis(traj[0], GM), a, rtol=1e-9)


# --- escape velocity -----------------------------------------------------------------------------


def test_escape_speed_formula_and_zero_energy():
    v_esc = ev_eq.escape_speed(ev_t.GM, ev_t.R)
    assert v_esc == pytest.approx(math.sqrt(2))
    y0 = ev_t.initial_state(torch.tensor([1.0], dtype=torch.float64), v_esc, 0.0, CPU)
    assert ev_eq.energy(y0, ev_t.GM).abs().item() < 1e-12
    assert ev_eq.dark_star_radius(1.327e20, 2.998e8) == pytest.approx(2953, rel=1e-3)  # the Sun: about 3 km


def test_energy_sign_decides_escape():
    f = torch.tensor([0.6, 0.8, 0.9, 1.1, 1.3], dtype=torch.float64)
    v_esc = ev_eq.escape_speed(ev_t.GM, ev_t.R)
    y0 = ev_t.initial_state(f, v_esc, 0.0, CPU)
    _, traj = ev_solve.solve(y0, ev_t.GM, ev_t.R, t_max=40.0, dt=0.005)
    E = ev_eq.energy(y0, ev_t.GM)
    assert torch.equal(ev_solve.escaped(traj, ev_t.R), E > 0)


def test_max_height_matches_energy_conservation():
    f = torch.tensor([0.5, 0.7, 0.85], dtype=torch.float64)
    v_esc = ev_eq.escape_speed(ev_t.GM, ev_t.R)
    y0 = ev_t.initial_state(f, v_esc, 0.0, CPU)
    _, traj = ev_solve.solve(y0, ev_t.GM, ev_t.R, t_max=20.0, dt=0.002)
    expected = ev_eq.max_radius(f * v_esc, ev_t.GM, ev_t.R)
    assert torch.allclose(ev_solve.highest_point(traj), expected, rtol=1e-4)


# --- Hubble expansion ----------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["matter", "de Sitter"])
def test_hubble_law_for_every_observer(model):
    H0 = hb_t.hubble_constant(70.0)
    q = hb_t.galaxy_grid(5, 500.0, CPU).double()
    t = torch.linspace(hb_eq.time_of(0.5, model, H0), hb_eq.time_of(1.5, model, H0), 50, dtype=torch.float64)
    for observer in range(len(q)):
        sol = hb_solve.solve(q, t, model, H0, observer)
        # v_rel = H d holds galaxy by galaxy, not just on average
        assert torch.allclose(sol["v_rel"], sol["H"][:, None, None] * sol["x_rel"], atol=1e-9)
        assert torch.allclose(hb_solve.fitted_hubble_rate(sol["x_rel"], sol["v_rel"]), sol["H"], rtol=1e-9)


@pytest.mark.parametrize("model", ["matter", "de Sitter"])
def test_hubble_rate_is_adot_over_a(model):
    H0 = hb_t.hubble_constant(70.0)
    t = torch.linspace(5.0, 20.0, 2001, dtype=torch.float64)
    a = hb_eq.scale_factor(t, model, H0)
    a_dot = torch.gradient(a, spacing=(t,))[0]
    H = hb_eq.hubble_rate(t, model, H0)
    assert torch.allclose((a_dot / a)[1:-1], H[1:-1], rtol=1e-5)
    assert hb_eq.scale_factor(torch.tensor(hb_eq.today(model, H0)), model, H0) == pytest.approx(1.0)


def test_redshift_is_ratio_of_scale_factors():
    assert hb_eq.redshift(torch.tensor(1.5), torch.tensor(0.5)).item() == pytest.approx(2.0)


# --- special relativity ----------------------------------------------------------------------------


def test_interval_invariant_under_boosts():
    eta = mk_eq.metric()
    events = torch.randn(50, 2, dtype=torch.float64)
    for beta in (-0.9, -0.3, 0.5, 0.99):
        boosted = mk_eq.transform(mk_eq.boost(torch.tensor(beta)), events)
        assert torch.allclose(mk_eq.interval(boosted, eta), mk_eq.interval(events, eta), atol=1e-9)


def test_boost_composition_is_velocity_addition():
    b1, b2 = torch.tensor(0.6, dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64)
    combined = mk_eq.boost(b1) @ mk_eq.boost(b2)
    assert torch.allclose(combined, mk_eq.boost(mk_eq.velocity_addition(b1, b2)), atol=1e-12)
    assert mk_eq.velocity_addition(b1, b2) < 1


def test_twin_paradox_ages():
    v, L = 0.8, 4.0
    home, traveller = mk_t.twin_world_lines(v, L)
    sol = mk_solve.solve(home, traveller, mk_t.frame_grid(1, -1, 1), beta_view=0.3)
    assert sol["tau_home"][-1].item() == pytest.approx(2 * L / v)
    assert sol["tau_trav"][-1].item() == pytest.approx(2 * L / v * math.sqrt(1 - v**2), rel=1e-9)


# --- Lagrange points -------------------------------------------------------------------------------


@pytest.mark.parametrize("mu", [0.001, 0.01, 0.03])
def test_lagrange_points_are_equilibria_and_L4_L5_equilateral(mu):
    points = lp_solve.lagrange_points(lp_t.lagrange_guesses(mu, CPU), mu)
    assert lp_eq.grad_potential(points, mu).abs().max() < 1e-12
    star = torch.tensor([-mu, 0.0, 0.0], dtype=torch.float64)
    planet = torch.tensor([1 - mu, 0.0, 0.0], dtype=torch.float64)
    for k in (3, 4):
        assert (points[k] - star).norm().item() == pytest.approx(1.0, abs=1e-10)
        assert (points[k] - planet).norm().item() == pytest.approx(1.0, abs=1e-10)
    # a particle at rest at any Lagrange point feels no acceleration in the rotating frame
    acc = lp_eq.acceleration(points, torch.zeros_like(points), mu, lp_eq.levi_civita())
    assert acc.abs().max() < 1e-12


def test_lagrange_jacobi_constant_conserved_and_L4_libration_bounded():
    mu = 0.01
    y0 = lp_t.particles_near_l4(mu, [3.0, 10.0], CPU)
    _, traj = lp_solve.solve(y0, mu, t_max=60.0, dt=0.02)
    C = lp_eq.jacobi_constant(traj, mu)
    assert (C - C[0]).abs().max() < 1e-8
    angle = torch.rad2deg(torch.atan2(traj[:, :, 0, 1], traj[:, :, 0, 0] + mu))
    assert angle.min() > 30 and angle.max() < 90  # stays near L4 at 60°


# --- tidal tensor ----------------------------------------------------------------------------------


def test_tidal_tensor_traceless_with_known_eigenvalues():
    x = torch.randn(20, 3, dtype=torch.float64) * 10
    T = td_eq.tidal_tensor(x, 1.0)
    r3 = x.norm(dim=-1) ** 3
    assert torch.allclose(torch.einsum("pii->p", T), torch.zeros(20, dtype=torch.float64), atol=1e-14)
    ev = torch.linalg.eigvalsh(T) * r3[:, None]
    expected = torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64).expand_as(ev)
    assert torch.allclose(ev, expected, atol=1e-9)


def test_tidal_tensor_autograd_matches_formula():
    x = torch.randn(10, 3, dtype=torch.float64) * 5
    assert torch.allclose(td_eq.tidal_tensor_autograd(x, 2.0), td_eq.tidal_tensor(x, 2.0), atol=1e-12)


def test_tidal_ring_stretches_radially_and_matches_linear_law_when_small():
    y0 = td_t.initial_state(40.0, 0.05, 16, CPU)
    sol = td_solve.solve(y0, 1.0, r_stop=10.0, n_steps=1500)
    along, across = td_solve.stretch(sol["xi_exact"])
    assert along[-1] > 2 * along[0] and across[-1] < 0.5 * across[0]
    assert torch.allclose(sol["xi_exact"][-1], sol["xi_tidal"][-1], atol=0.02 * along[-1].item())
    # transverse squeeze is exactly r/r0 for a radial fall (all particles head for the same point)
    assert across[-1].item() / 0.05 == pytest.approx(10.0 / 40.0, rel=1e-3)


# --- black hole geodesics --------------------------------------------------------------------------


def test_autograd_christoffels_match_schwarzschild():
    M, r, th = 1.0, 7.3, 1.1
    x = torch.tensor([[0.4, r, th, 0.9]], dtype=torch.float64)
    G = bh_eq.christoffel(x, M)[0]
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
    _, traj = bh_solve.geodesic(bh_t.bound_orbit(14.0, 0.4), 1.0, 600.0, 2.0)
    assert (bh_eq.norm(traj, 1.0) + 1).abs().max() < 1e-6


def test_weak_field_perihelion_precession():
    p, e = 100.0, 0.3
    a = p / (1 - e**2)
    _, traj = bh_solve.geodesic(bh_t.bound_orbit(p, e), 1.0, 1.3 * 2 * math.pi * a**1.5, 10.0)
    advance = bh_solve.periapsis_angles(traj).diff()[0] - 2 * math.pi
    assert advance.item() == pytest.approx(bh_eq.weak_field_precession(p, 1.0), rel=0.1)


def test_isco_and_photon_sphere_radii():
    M = 1.0
    r = torch.linspace(3.5, 20, 20001, dtype=torch.float64)
    # at L² = 12 M² the minimum and maximum of V merge: dV/dr and d²V/dr² vanish together at r = 6M
    V = bh_eq.effective_potential(r, math.sqrt(12) * M, M)
    dV = torch.gradient(V, spacing=(r,))[0]
    assert r[dV.abs().argmin()].item() == pytest.approx(bh_eq.isco_radius(M), rel=1e-2)
    # light: V_light ∝ (1 − 2M/r)/r² peaks at the photon sphere
    rr = torch.linspace(2.1, 10, 20001, dtype=torch.float64)
    V_light = (1 - 2 * M / rr) / rr**2
    assert rr[V_light.argmax()].item() == pytest.approx(bh_eq.photon_sphere_radius(M), rel=1e-3)


def test_radial_infall_reaches_horizon_in_finite_proper_time():
    r0 = 10.0
    tau, traj = bh_solve.infall(bh_t.radial_infall(r0), 1.0, 0.02)
    r, t = traj[:, 0, 0, 1], traj[:, 0, 0, 0]
    assert r[-1].item() == pytest.approx(2.1, abs=0.02)
    assert tau[-1].item() < bh_solve.proper_infall_time(r0, 2.0, 1.0) < 40  # finite
    assert t[-1] > tau[-1]  # the far-away clock runs ahead: t → ∞ at the horizon
    assert bh_solve.proper_infall_time(r0, 0.0, 1.0) == pytest.approx(math.pi / 2 * math.sqrt(r0**3 / 2))


# --- Hawking radiation -----------------------------------------------------------------------------


def test_hawking_temperature_of_the_sun_and_inverse_mass_law():
    M = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float64) * hr_eq.M_SUN
    T = hr_eq.hawking_temperature(M)
    assert T[0].item() == pytest.approx(6.17e-8, rel=5e-3)
    assert torch.allclose(T * M, (T * M)[0].expand(3), rtol=1e-12)


def test_wien_peak_of_planck_spectrum():
    T = torch.tensor([1e3, 1e9], dtype=torch.float64)
    lam = torch.logspace(-16, -2, 200001, dtype=torch.float64)
    B = hr_eq.planck(lam, T)  # (2, L)
    peak = lam[B.argmax(dim=1)]
    assert torch.allclose(peak * T, torch.full_like(T, hr_eq.WIEN_B), rtol=1e-3)


def test_integrated_lifetime_matches_formula():
    M0 = torch.tensor([1e5, 1e11], dtype=torch.float64)
    times, masses = hr_solve.evaporate(M0, fraction=0.005)
    assert torch.allclose(times[-1], hr_eq.lifetime(M0), rtol=1e-6)
    # M³ falls linearly in time: t = t_life(M0) − t_life(M) at every step
    remaining = hr_eq.lifetime(M0) - hr_eq.lifetime(masses)
    assert ((times - remaining).abs() / hr_eq.lifetime(M0)).max() < 1e-6


def test_power_scales_as_inverse_mass_squared():
    M = torch.tensor([1e10, 2e10], dtype=torch.float64)
    P = hr_eq.power(M)
    assert (P[0] / P[1]).item() == pytest.approx(4.0)
    expected = hr_eq.HBAR * hr_eq.C**6 / (15360 * math.pi * hr_eq.G**2 * M**2)
    assert torch.allclose(P, expected, rtol=1e-6)


# --- gravitational lensing -------------------------------------------------------------------------


def test_null_geodesic_deflection_approaches_einstein_value():
    b = torch.tensor([300.0, 1000.0], dtype=torch.float64)
    y0 = torch.stack([torch.zeros_like(b), 1 / b], dim=1)
    rays = gl_solve.trace_rays(y0, 1.0, dphi=0.002, phi_max=3.3)
    alpha = gl_solve.deflection_angles(rays)
    assert torch.allclose(alpha, gl_eq.thin_lens_deflection(b, 1.0), rtol=0.015)


def test_rays_inside_shadow_are_captured():
    b, y0 = gl_t.ray_fan(4.0, 6.0, 5, CPU)
    alpha = gl_solve.deflection_angles(gl_solve.trace_rays(y0, 1.0))
    captured = torch.isnan(alpha)
    assert torch.equal(captured, b < gl_eq.shadow_radius(1.0))


def test_einstein_radius_maps_to_source_on_axis():
    GM_c2, D_l, D_s = 1477.0 * 10, 1.2e20, 2.5e20  # 10 solar masses, metres
    theta_E = gl_eq.einstein_radius(GM_c2, D_l, D_s)
    alpha = 4 * GM_c2 / (D_l * theta_E)  # deflection of the ray passing at b = D_l θ_E
    beta = theta_E - (D_s - D_l) / D_s * alpha  # lens equation in physical angles
    assert abs(beta) < 1e-12 * theta_E
    # and in θ_E units the ring θ = 1 maps onto the source position β = 0
    ring = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    assert torch.allclose(gl_eq.lens_map(ring), torch.zeros(2, 2), atol=1e-7)


def test_image_plane_magnification_matches_point_lens():
    theta = gl_t.sky_grid(400, 2.5, CPU).double()
    u = torch.tensor([0.4, 0.8, 1.2], dtype=torch.float64)
    centres = torch.stack([u, torch.zeros_like(u)], dim=-1)
    lensed, plain = gl_solve.lensed_images(theta, centres, radius=0.03)
    assert torch.allclose(gl_solve.magnification(lensed, plain), gl_eq.point_magnification(u), rtol=0.02)
