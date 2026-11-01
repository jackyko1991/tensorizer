import math

import pytest
import torch

from problems._common.integrators import integrate
from problems.mechanics.coupled_oscillators import equation as co_eq
from problems.mechanics.coupled_oscillators import solve as co_solve
from problems.mechanics.coupled_oscillators import tensors as co_t
from problems.mechanics.double_pendulum import equation as dp_eq
from problems.mechanics.double_pendulum import solve as dp_solve
from problems.mechanics.double_pendulum import tensors as dp_t
from problems.mechanics.inclined_plane import equation as ip_eq
from problems.mechanics.inclined_plane import solve as ip_solve
from problems.mechanics.inclined_plane import tensors as ip_t
from problems.mechanics.lagrangian_autograd import equation as la_eq
from problems.mechanics.lagrangian_autograd import solve as la_solve
from problems.mechanics.lagrangian_autograd import tensors as la_t
from problems.mechanics.mass_spring_cloth import equation as cl_eq
from problems.mechanics.mass_spring_cloth import solve as cl_solve
from problems.mechanics.mass_spring_cloth import tensors as cl_t
from problems.mechanics.n_body_gravity import equation as nb_eq
from problems.mechanics.n_body_gravity import solve as nb_solve
from problems.mechanics.n_body_gravity import tensors as nb_t
from problems.mechanics.projectile import equation as proj_eq
from problems.mechanics.projectile import solve as proj_solve
from problems.mechanics.projectile import tensors as proj_t
from problems.mechanics.rigid_body_rotation import equation as rb_eq
from problems.mechanics.rigid_body_rotation import solve as rb_solve
from problems.mechanics.rigid_body_rotation import tensors as rb_t
from problems.mechanics.simple_pendulum import equation as sp_eq
from problems.mechanics.simple_pendulum import solve as sp_solve
from problems.mechanics.simple_pendulum import tensors as sp_t

CPU = torch.device("cpu")


def test_projectile_without_drag_matches_textbook_range():
    g = proj_t.gravity(CPU)
    angles = proj_t.launch_angles(5)
    y0 = proj_t.initial_state(20.0, angles, CPU)
    _, traj = proj_solve.solve(y0, g, k=0.0, t_max=5.0, dt=0.001)
    expected = 20.0**2 * torch.sin(2 * angles) / 9.81
    assert torch.allclose(proj_solve.ranges(traj), expected, rtol=5e-3)


def test_projectile_energy_conserved_without_drag():
    g = proj_t.gravity(CPU)
    y0 = proj_t.initial_state(15.0, torch.tensor([math.radians(60.0)]), CPU)
    _, traj = proj_solve.solve(y0, g, k=0.0, t_max=1.0)  # still airborne
    e = proj_eq.energy(traj, g)
    assert torch.allclose(e, e[0].expand_as(e), rtol=1e-4)  # float32 round-off


def test_projectile_drag_shortens_range_and_lowers_best_angle():
    g = proj_t.gravity(CPU)
    angles = proj_t.launch_angles(13)
    y0 = proj_t.initial_state(30.0, angles, CPU)
    _, traj = proj_solve.solve(y0, g, k=0.05, t_max=8.0)
    r = proj_solve.ranges(traj)
    assert r.max() < 30.0**2 / 9.81
    assert angles[r.argmax()] < math.radians(45.0)


# --- coupled oscillators ---


def test_coupled_oscillator_frequencies_match_analytic():
    for n in (2, 5, 10):
        K = co_t.stiffness(n, k=4.0, device=CPU)
        omega, e = co_eq.normal_modes(K, m=0.5)
        assert torch.allclose(omega, co_t.analytic_frequencies(n, k=4.0, m=0.5), rtol=1e-5)
        assert torch.allclose(e.T @ e, torch.eye(n), atol=1e-5)  # orthonormal modes


def test_coupled_oscillator_energy_conserved():
    K = co_t.stiffness(6, k=10.0, device=CPU)
    x0 = co_t.initial_displacement(6, "first mass", 0.3, CPU)
    sol = co_solve.solve(K, 1.0, x0, t_max=20.0)
    kinetic, potential = co_eq.energy(sol["x"], sol["v"], K, 1.0)
    total = kinetic + potential
    assert torch.allclose(total, total[0].expand_as(total), rtol=1e-5)
    assert torch.allclose(total[0], potential[0])  # starts at rest


def test_coupled_oscillator_modes_match_direct_integration():
    """The normal-mode superposition equals a brute-force RK4 solution of m ẍ_i = -K_ij x_j."""
    K, m = co_t.stiffness(4, k=3.0, device=CPU), 2.0
    x0 = co_t.initial_displacement(4, "middle mass", 0.2, CPU)
    sol = co_solve.solve(K, m, x0, t_max=5.0, n_times=501)  # dt = 0.01

    def rhs(y, t):
        return torch.stack([y[1], co_eq.spring_force(K, y[0]) / m])

    traj = integrate(rhs, torch.stack([x0, torch.zeros_like(x0)]), 0.01, 500)
    assert torch.allclose(traj[:, 0], sol["x"], atol=1e-4)


def test_coupled_oscillator_single_mode_keeps_its_shape():
    K = co_t.stiffness(5, k=1.0, device=CPU)
    x0 = co_t.initial_displacement(5, "mode 2", 0.3, CPU)
    sol = co_solve.solve(K, 1.0, x0, t_max=10.0)
    assert (sol["A"] > 1e-4).sum() == 1  # only one mode is excited
    ratio = sol["x"][:, 0] / x0[0]  # every mass follows the same cos(ω t)
    assert torch.allclose(sol["x"], ratio[:, None] * x0, atol=1e-5)


# --- N-body gravity ---


def test_n_body_energy_and_momentum_conserved():
    for scenario in ("figure-8", "star and planets", "binary and planet"):
        m, x0, v0 = nb_t.initial_state(scenario, CPU)
        _, xs, vs = nb_solve.solve(m, x0, v0, nb_t.G, eps=0.0, t_max=6.0, dt=0.002)
        e0 = nb_eq.energy(xs[0], vs[0], m, nb_t.G, 0.0)
        e1 = nb_eq.energy(xs[-1], vs[-1], m, nb_t.G, 0.0)
        assert abs((e1 - e0) / e0) < 1e-5, scenario
        p = nb_eq.momentum(vs, m)  # (T, 3)
        assert p.abs().max() < 1e-10, scenario


def test_n_body_two_body_period_matches_kepler():
    m1, m2, a = 1.0, 0.3, 1.5
    m, x0, v0 = nb_t.two_body_circular(m1, m2, a, CPU)
    period = 2 * math.pi * math.sqrt(a**3 / (nb_t.G * (m1 + m2)))
    times, xs, _ = nb_solve.solve(m, x0, v0, nb_t.G, eps=0.0, t_max=1.5 * period, dt=0.001)
    rel = xs[:, 1] - xs[:, 0]  # (T, 3) separation vector
    angle = torch.atan2(rel[:, 1], rel[:, 0])
    step = (angle.diff() + math.pi) % (2 * math.pi) - math.pi  # angle swept per step, no 2π jumps
    swept = torch.cat([torch.zeros(1, dtype=step.dtype), torch.cumsum(step, 0)])  # (T,) increasing
    measured = times[torch.searchsorted(swept, torch.tensor([2 * math.pi], dtype=swept.dtype))].item()
    assert measured == pytest.approx(period, rel=1e-3)
    assert torch.allclose(rel.norm(dim=-1), torch.tensor(a, dtype=rel.dtype), rtol=1e-6)  # stays circular


def test_n_body_figure_eight_is_periodic():
    m, x0, v0 = nb_t.figure_eight(CPU)
    _, xs, _ = nb_solve.solve(m, x0, v0, nb_t.G, eps=0.0, t_max=6.32591398, dt=0.001)
    assert torch.allclose(xs[-1], x0, atol=1e-3)


# --- simple pendulum ---


def test_simple_pendulum_small_amplitude_period():
    g, L = 9.81, 1.5
    y0 = sp_t.initial_state(torch.deg2rad(torch.tensor([2.0])), CPU)
    times, traj = sp_solve.solve(y0, g, L, t_max=3.0, dt=0.001)
    period = sp_solve.periods(times, traj[:, :, 0])
    assert period.item() == pytest.approx(2 * math.pi * math.sqrt(L / g), rel=1e-3)


def test_simple_pendulum_large_amplitude_is_slower_and_matches_elliptic_integral():
    g, L = 9.81, 1.0
    theta0 = torch.deg2rad(torch.tensor([30.0, 90.0, 150.0]))
    times, traj = sp_solve.solve(sp_t.initial_state(theta0, CPU), g, L, t_max=4.0, dt=0.001)
    measured = sp_solve.periods(times, traj[:, :, 0])
    small = 2 * math.pi * math.sqrt(L / g)
    assert (measured > small).all() and (measured.diff() > 0).all()
    assert torch.allclose(measured, sp_eq.exact_period(theta0, g, L), rtol=2e-3)
    # series check at 90°: T ≈ T0 (1 + θ²/16 + 11θ⁴/3072)
    series = small * (1 + (math.pi / 2) ** 2 / 16 + 11 * (math.pi / 2) ** 4 / 3072)
    assert measured[1].item() == pytest.approx(series, rel=0.01)


def test_simple_pendulum_energy_conserved():
    y0 = sp_t.initial_state(sp_t.amplitudes(4, 160.0), CPU)
    _, traj = sp_solve.solve(y0, 9.81, 1.0, t_max=6.0)
    e = sp_eq.energy(traj, 9.81, 1.0)  # (T, B)
    assert torch.allclose(e, e[0].expand_as(e), rtol=1e-4, atol=1e-5)


def test_simple_pendulum_small_angle_solution_matches_for_tiny_swing():
    y0 = sp_t.initial_state(torch.deg2rad(torch.tensor([1.0])), CPU)
    times, traj = sp_solve.solve(y0, 9.81, 1.0, t_max=4.0)
    small = sp_eq.small_angle(y0[:, 0], times, 9.81, 1.0)
    assert (traj[:, :, 0] - small).abs().max() < 0.01 * y0[0, 0]


# --- inclined plane ---


def test_inclined_plane_sliding_acceleration():
    g, mu_s, mu_k = 9.81, 0.3, 0.2
    theta = torch.deg2rad(torch.tensor([20.0, 35.0, 60.0]))  # tan θ = 0.36, 0.70, 1.73 > μ_s
    s, n = ip_t.unit_vectors(theta, CPU)
    a = ip_eq.acceleration(ip_t.gravity(g, CPU), s, n, mu_s, mu_k)
    expected = g * (torch.sin(theta) - mu_k * torch.cos(theta))
    assert torch.allclose(torch.einsum("bi,bi->b", a, s), expected, atol=1e-5)
    assert torch.allclose(torch.einsum("bi,bi->b", a, n), torch.zeros(3), atol=1e-5)  # stays on the surface


def test_inclined_plane_stays_at_rest_below_friction_angle():
    g, mu_s, mu_k = 9.81, 0.6, 0.4
    theta = torch.deg2rad(torch.tensor([5.0, 20.0, 30.0]))  # all tan θ < 0.6
    s, n = ip_t.unit_vectors(theta, CPU)
    a = ip_eq.acceleration(ip_t.gravity(g, CPU), s, n, mu_s, mu_k)
    assert torch.allclose(a, torch.zeros_like(a), atol=1e-5)
    sol = ip_solve.solve(a, s, ramp_length=3.0, t_max=2.0)
    assert sol["d"].abs().max() < 1e-4


def test_inclined_plane_frictionless_distance():
    theta = torch.deg2rad(torch.tensor([30.0]))
    s, n = ip_t.unit_vectors(theta, CPU)
    a = ip_eq.acceleration(ip_t.gravity(9.81, CPU), s, n, 0.0, 0.0)
    sol = ip_solve.solve(a, s, ramp_length=100.0, t_max=2.0)
    assert sol["d"][-1, 0].item() == pytest.approx(0.5 * 9.81 * 0.5 * 4.0, rel=1e-5)  # ½ g sin30° t²


# --- double pendulum ---


def test_double_pendulum_energy_conserved():
    params = (1.0, 2.0, 1.0, 0.7, 9.81)
    y0 = dp_t.initial_state(120.0, -10.0, 0.01, CPU)
    _, traj = dp_solve.solve(y0, *params, t_max=10.0, dt=0.002)
    e = dp_eq.energy(traj, *params)  # (T, B)
    scale = (params[0] + params[1]) * 9.81 * (params[2] + params[3])  # typical energy size
    assert (e - e[0]).abs().max() < 1e-5 * scale


def test_double_pendulum_mass_matrix_solve_matches_explicit_formula():
    """torch.linalg.solve on M_ij q̈_j = f_i equals the textbook closed-form angular accelerations."""
    m1, m2, L1, L2, g = 1.3, 0.8, 1.1, 0.9, 9.81
    t1, t2, w1, w2 = 0.7, -1.2, 0.5, -2.0
    y = torch.tensor([[[t1, t2], [w1, w2]]], dtype=torch.float64)
    d = t1 - t2
    den = 2 * m1 + m2 - m2 * math.cos(2 * d)
    a1 = (
        -g * (2 * m1 + m2) * math.sin(t1)
        - m2 * g * math.sin(t1 - 2 * t2)
        - 2 * math.sin(d) * m2 * (w2**2 * L2 + w1**2 * L1 * math.cos(d))
    ) / (L1 * den)
    a2 = 2 * math.sin(d) * (w1**2 * L1 * (m1 + m2) + g * (m1 + m2) * math.cos(t1) + w2**2 * L2 * m2 * math.cos(d))
    a2 /= L2 * den
    q_ddot = dp_eq.rhs(y, m1, m2, L1, L2, g)[0, 1]
    assert torch.allclose(q_ddot, torch.tensor([a1, a2], dtype=torch.float64), atol=1e-10)


def test_double_pendulum_small_angle_normal_modes():
    g, L = 9.81, 1.0
    omega, e = dp_eq.small_angle_modes(1.0, 1.0, L, L, g)
    analytic = torch.tensor([(2 - math.sqrt(2)) * g / L, (2 + math.sqrt(2)) * g / L], dtype=torch.float64).sqrt()
    assert torch.allclose(omega, analytic, rtol=1e-10)
    # start in the slow mode with a tiny amplitude: the full nonlinear motion is a pure cosine at ω_1
    amp = 1e-3
    q0 = amp * e[:, 0] / e[0, 0]
    y0 = torch.stack([q0, torch.zeros(2, dtype=torch.float64)])[None]  # (1, 2, 2)
    times, traj = dp_solve.solve(y0, 1.0, 1.0, L, L, g, t_max=5.0, dt=0.002)
    expected = amp * torch.cos(omega[0] * times)
    assert (traj[:, 0, 0, 0] - expected).abs().max() < 0.01 * amp


def test_double_pendulum_is_chaotic():
    y0 = dp_t.initial_state(120.0, -10.0, 1e-4, CPU)
    _, traj = dp_solve.solve(y0, 1.0, 1.0, 1.0, 1.0, 9.81, t_max=20.0)
    gap = (traj[:, 0, 0] - traj[:, 1, 0]).norm(dim=-1)
    assert gap[0] < 1e-5 and gap[-1] > 0.1


# --- rigid body rotation ---


def test_rigid_body_energy_and_angular_momentum_conserved():
    inertia = rb_t.box_inertia(2.0, 3.0, 2.0, 1.0)
    y0 = rb_t.initial_state("intermediate", 5.0, 0.01)
    sol = rb_solve.solve(y0, inertia, t_max=10.0)
    T = rb_eq.kinetic_energy(sol["omega"], inertia)
    L_body = rb_eq.angular_momentum(sol["omega"], inertia)
    L_sq = torch.einsum("ti,ti->t", L_body, L_body)
    assert torch.allclose(T, T[0].expand_as(T), rtol=1e-8)
    assert torch.allclose(L_sq, L_sq[0].expand_as(L_sq), rtol=1e-8)
    L_world = rb_eq.world_angular_momentum(sol["R"], sol["omega"], inertia)  # (T, 3) fixed in space
    assert torch.allclose(L_world, L_world[0].expand_as(L_world), atol=1e-6)


def test_rigid_body_major_and_minor_axes_stable_intermediate_flips():
    inertia = rb_t.box_inertia(1.0, 3.0, 2.0, 1.0)
    for axis in ("major", "minor"):
        w = rb_solve.solve(rb_t.initial_state(axis, 5.0, 0.01), inertia, t_max=15.0)["omega"]
        main = rb_t.AXES[axis]
        others = [i for i in range(3) if i != main]
        assert w[:, main].min() > 0.99 * 5.0, axis  # spin barely changes
        assert w[:, others].abs().max() < 0.2, axis  # wobble stays small
    w = rb_solve.solve(rb_t.initial_state("intermediate", 5.0, 0.01), inertia, t_max=15.0)["omega"]
    assert w[:, 1].min() < -4.5  # the intermediate-axis spin reverses: the box flipped over


def test_rigid_body_euler_equations_match_component_form():
    """The ε-einsum equals the textbook (I_2 − I_3) ω_2 ω_3 / I_1 etc. for a diagonal inertia tensor."""
    inertia = rb_t.box_inertia(1.0, 3.0, 2.0, 1.0)
    I1, I2, I3 = inertia.diagonal().tolist()
    w = torch.tensor([0.3, -1.2, 2.0], dtype=torch.float64)
    w_dot = torch.linalg.solve(inertia, rb_eq.gyroscopic_torque(w, inertia, rb_eq.levi_civita()))
    expected = torch.stack([(I2 - I3) * w[1] * w[2] / I1, (I3 - I1) * w[2] * w[0] / I2, (I1 - I2) * w[0] * w[1] / I3])
    assert torch.allclose(w_dot, expected, atol=1e-12)


# --- Lagrangian mechanics by autograd ---


def test_autograd_harmonic_oscillator_and_pendulum_match_hand_derived():
    m, k, g, ell = 2.0, 8.0, 9.81, 1.3

    def oscillator(q, q_dot):
        return 0.5 * m * (q_dot**2).sum(-1) - 0.5 * k * (q**2).sum(-1)

    def pendulum(q, q_dot):
        return 0.5 * m * ell**2 * q_dot[..., 0] ** 2 + m * g * ell * torch.cos(q[..., 0])

    for x in (0.3, -1.1, 2.0):
        q = torch.tensor([x], dtype=torch.float64)
        q_dot = torch.tensor([0.7], dtype=torch.float64)
        assert la_eq.euler_lagrange(oscillator, q, q_dot).item() == pytest.approx(-k / m * x, rel=1e-12)
        assert la_eq.euler_lagrange(pendulum, q, q_dot).item() == pytest.approx(-g / ell * math.sin(x), rel=1e-12)


def test_autograd_spring_pendulum_matches_hand_derived():
    m, k, ell0, g = 1.5, 30.0, 1.0, 9.81
    L = la_eq.spring_pendulum(m, k, ell0, g)
    for r, th, r_dot, th_dot in [(1.3, 0.4, 0.2, -1.0), (0.8, -2.0, -1.5, 3.0)]:
        q = torch.tensor([r, th], dtype=torch.float64)
        q_dot = torch.tensor([r_dot, th_dot], dtype=torch.float64)
        r_ddot = r * th_dot**2 + g * math.cos(th) - k / m * (r - ell0)
        th_ddot = (-g * math.sin(th) - 2 * r_dot * th_dot) / r
        got = la_eq.euler_lagrange(L, q, q_dot)
        assert torch.allclose(got, torch.tensor([r_ddot, th_ddot], dtype=torch.float64), atol=1e-12)


def test_autograd_spring_pendulum_energy_conserved():
    L = la_eq.spring_pendulum(1.0, 30.0, 1.0, 9.81)
    sol = la_solve.solve(la_t.initial_state(1.4, 30.0), L, t_max=5.0)
    E = sol["E"]
    assert ((E - E[0]) / E[0]).abs().max() < 1e-6
    # the autograd energy q̇·∂L/∂q̇ − L equals the hand-written T + V
    r, th = sol["q"][:, 0], sol["q"][:, 1]
    r_dot, th_dot = sol["q_dot"][:, 0], sol["q_dot"][:, 1]
    by_hand = 0.5 * (r_dot**2 + r**2 * th_dot**2) - 9.81 * r * torch.cos(th) + 15.0 * (r - 1.0) ** 2
    assert torch.allclose(E, by_hand, atol=1e-12)


# --- mass-spring cloth ---


def hanging_chain(n: int, spacing: float):
    """A vertical chain of n nodes hanging from node 0: positions, D, rest lengths, free mask."""
    x0 = torch.zeros(n, 3, dtype=torch.float64)
    x0[:, 2] = -spacing * torch.arange(n, dtype=torch.float64)
    edges = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=-1)
    free = torch.ones(n, dtype=torch.float64)
    free[0] = 0.0
    return x0, cl_t.incidence(edges, n), cl_t.rest_lengths(x0, edges), free


def test_cloth_chain_static_equilibrium_matches_analytic_stretch():
    n, m, k, g = 6, 0.1, 50.0, 9.81
    x0, D, rest, free = hanging_chain(n, 0.2)
    cloth = cl_t.parameters(k, gamma=4.0, node_mass=m, g=g)
    _, traj = cl_solve.solve(x0, D, rest, free, cloth, t_max=15.0, dt=0.005)
    x_end = traj[-1, 0]
    stretch = (x_end[1:] - x_end[:-1]).norm(dim=-1) - rest  # (n-1,)
    below = torch.arange(n - 1, 0, -1, dtype=torch.float64)  # spring e carries the n-1-e masses under it
    assert torch.allclose(stretch, below * m * g / k, rtol=1e-4)
    assert traj[-1, 1].abs().max() < 1e-5  # at rest


def test_cloth_einsum_forces_match_index_add():
    n = 5
    x0 = cl_t.grid(n, 0.1)
    edges = cl_t.springs(n)
    D, rest = cl_t.incidence(edges, n * n), cl_t.rest_lengths(x0, edges)
    x = x0 + 0.02 * torch.randn(n * n, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    F1 = cl_eq.spring_forces(x, D, rest, 100.0)
    F2 = cl_eq.spring_forces_scatter(x, edges, rest, 100.0)
    assert torch.allclose(F1, F2, atol=1e-12)
    assert torch.allclose(F1.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-12)  # internal forces cancel


def test_cloth_energy_decreases_with_damping_and_is_conserved_without():
    n, m, k = 6, 0.01, 100.0
    x0 = cl_t.grid(n, 0.1)
    edges = cl_t.springs(n)
    D, rest, free = cl_t.incidence(edges, n * n), cl_t.rest_lengths(x0, edges), cl_t.free_mask(n)

    damped = cl_t.parameters(k, gamma=1.0, node_mass=m, g=9.81)
    _, traj = cl_solve.solve(x0, D, rest, free, damped, t_max=1.5, dt=0.001)
    E = cl_eq.energy(traj, D, rest, damped)
    assert (E.diff() <= 1e-9).all()  # never goes up
    assert E[-1] < E[0] - 0.05

    undamped = cl_t.parameters(k, gamma=0.0, node_mass=m, g=9.81)
    _, traj = cl_solve.solve(x0, D, rest, free, undamped, t_max=0.5, dt=0.0005)
    E = cl_eq.energy(traj, D, rest, undamped)
    kinetic_max = 0.5 * m * torch.einsum("tai,tai->t", traj[:, 1], traj[:, 1]).max()
    assert (E - E[0]).abs().max() < 1e-4 * kinetic_max
