import math

import torch

from problems.em.biot_savart import equation as bs_eq
from problems.em.biot_savart import tensors as bs_t
from problems.em.coulomb_field import equation as cf_eq
from problems.em.electrostatics import equation as es_eq
from problems.em.electrostatics import solve as es_solve
from problems.em.electrostatics import tensors as es_t
from problems.em.em_wave_fdtd import equation as fd_eq
from problems.em.em_wave_fdtd import solve as fd_solve
from problems.em.em_wave_fdtd import tensors as fd_t
from problems.em.field_tensor_boost import equation as ft_eq
from problems.em.field_tensor_boost import solve as ft_solve
from problems.em.field_tensor_boost import tensors as ft_t
from problems.em.lorentz_force import equation as lf_eq
from problems.em.lorentz_force import solve as lf_solve
from problems.em.lorentz_force import tensors as lf_t
from problems.em.rc_circuit import equation as rc_eq
from problems.em.rc_circuit import solve as rc_solve
from problems.em.rc_circuit import tensors as rc_t
from problems.em.relativistic_particle import equation as rp_eq
from problems.em.relativistic_particle import solve as rp_solve
from problems.em.relativistic_particle import tensors as rp_t
from problems.em.waveguide_modes import equation as wg_eq
from problems.em.waveguide_modes import solve as wg_solve
from problems.em.waveguide_modes import tensors as wg_t

CPU = torch.device("cpu")


# --- Lorentz force -------------------------------------------------------------------------------


def test_levi_civita_cross_product_matches_torch():
    eps = lf_t.levi_civita(CPU)
    v = torch.randn(7, 3)
    B = torch.randn(3)
    expected = torch.linalg.cross(v, B.expand_as(v))
    assert torch.allclose(lf_eq.cross(eps, v, B), expected, atol=1e-6)


def test_cyclotron_period_and_radius():
    eps = lf_t.levi_civita(CPU)
    E, B = lf_t.fields(0.0, 1.5, CPU)
    q, m = lf_t.particles(4, CPU)  # charges ±1, +2 and a heavier particle
    y0 = lf_t.initial_state(4, 0.0, CPU)  # |v⊥| = 1, no motion along B
    omega = lf_eq.cyclotron_frequency(q, m, B)
    assert torch.allclose(omega, q.abs() * 1.5 / m)

    period = 2 * math.pi / omega.min().item()  # longest period in the batch
    _, traj = lf_solve.solve(y0, q, m, E, B, eps, t_max=period, dt=period / 2000)
    pos = traj[:, :, 0, :]  # (T, P, 3)
    radius = 0.5 * (pos[..., 0].amax(0) - pos[..., 0].amin(0))  # half the x extent of each circle
    assert torch.allclose(radius, m * 1.0 / (q.abs() * 1.5), rtol=1e-3)
    # every particle completes a whole number of turns in the longest period, so all are back at the start
    assert torch.allclose(pos[-1], pos[0], atol=1e-4)


def test_exb_drift_velocity():
    eps = lf_t.levi_civita(CPU)
    E, B = lf_t.fields(0.4, 2.0, CPU)
    u = lf_eq.drift_velocity(E, B, eps)
    assert torch.allclose(u, torch.linalg.cross(E, B) / (B @ B))
    assert torch.allclose(u, torch.tensor([0.2, 0.0, 0.0]))  # E/B along x, for any charge

    q, m = lf_t.particles(4, CPU)
    y0 = lf_t.initial_state(4, 0.0, CPU)
    period = 2 * math.pi * 2.0 / 2.0  # 2π m/(|q| B) of the heaviest particle (m=2, q=1) — a multiple of the others
    _, traj = lf_solve.solve(y0, q, m, E, B, eps, t_max=period, dt=period / 2000)
    displacement = traj[-1, :, 0, :] - traj[0, :, 0, :]  # (P, 3) after whole gyrations
    assert torch.allclose(displacement / period, u.expand_as(displacement), atol=1e-3)


# --- Electrostatics ------------------------------------------------------------------------------


def test_discrete_laplacian_matches_analytic():
    n = 40
    x, h = es_t.grid(n)
    X, Y = torch.meshgrid(x, x, indexing="xy")  # rows = y, columns = x
    phi = torch.sin(math.pi * X) * torch.sin(math.pi * Y)  # vanishes on the walls
    lap = es_eq.apply_laplacian(es_eq.laplacian(n, h), phi)
    assert torch.allclose(lap, -2 * math.pi**2 * phi, atol=2e-2 * 2 * math.pi**2)


def test_poisson_solution_satisfies_equation():
    n = 30
    _, h = es_t.grid(n)
    rho = es_t.charge_density("quadrupole", n, 1.0)
    phi, _ = es_solve.solve(rho, h)
    assert torch.allclose(es_eq.apply_laplacian(es_eq.laplacian(n, h), phi), -rho, atol=1e-6 * rho.abs().max())


def test_gauss_law_flux_around_one_charge():
    n = 60
    _, h = es_t.grid(n)
    rho = es_t.charge_density("dipole", n, 1.0)  # +1 at x≈0.35, −1 at x≈0.65
    _, E = es_solve.solve(rho, h)
    a, b = (rho > 0).nonzero()[0].tolist()
    flux = es_solve.flux_through_square(E, h, a - 8, a + 8, b - 8, b + 8)  # box around the + charge only
    assert abs(flux.item() - 1.0) < 0.1
    whole = es_solve.flux_through_square(E, h, 2, n - 3, 2, n - 3)  # encloses both: net charge 0
    assert abs(whole.item()) < 0.1


# --- Field tensor under a boost ------------------------------------------------------------------


def make_F(E, B):
    E = torch.tensor(E, dtype=ft_t.DTYPE)
    B = torch.tensor(B, dtype=ft_t.DTYPE)
    return E, B, ft_eq.field_tensor(E, B, ft_t.levi_civita(3))


def test_field_tensor_round_trip_and_antisymmetry():
    E, B, F = make_F([0.3, -1.0, 0.7], [0.2, 0.5, -0.4])
    assert torch.allclose(F, -F.T)
    assert torch.allclose(ft_eq.electric(F), E)
    assert torch.allclose(ft_eq.magnetic(F, ft_t.levi_civita(3)), B)


def test_invariants_unchanged_by_boost():
    E, B, F = make_F([0.3, 1.0, 0.2], [0.4, -0.3, 0.5])
    sweep = ft_solve.solve(F, beta_max=0.95, n_steps=20)
    assert torch.allclose(sweep["FF"], 2 * (B @ B - E @ E).expand(20))
    assert torch.allclose(sweep["FFdual"], -4 * (E @ B).expand(20))
    E_dot_B = torch.einsum("si,si->s", sweep["E"], sweep["B"])
    assert torch.allclose(E_dot_B, (E @ B).expand(20))


def test_boost_of_pure_electric_field_creates_magnetic_field():
    E, _, F = make_F([0.5, 1.0, -0.3], [0.0, 0.0, 0.0])
    sweep = ft_solve.solve(F, beta_max=0.9, n_steps=10)
    beta_vec = torch.zeros(10, 3, dtype=ft_t.DTYPE)
    beta_vec[:, 0] = sweep["beta"]  # β along x
    gamma = 1 / torch.sqrt(1 - sweep["beta"] ** 2)
    expected_B = -gamma[:, None] * torch.linalg.cross(beta_vec, E.expand(10, 3))
    assert torch.allclose(sweep["B"], expected_B)
    # parallel E unchanged, transverse E stretched by γ
    assert torch.allclose(sweep["E"][:, 0], E[0].expand(10))
    assert torch.allclose(sweep["E"][:, 1:], gamma[:, None] * E[1:])


def test_boost_matrix_preserves_metric():
    Lam = ft_t.boost(torch.tensor([0.0, 0.5, 0.99], dtype=ft_t.DTYPE))
    eta = ft_t.metric()
    assert torch.allclose(torch.einsum("sma,snb,mn->sab", Lam, Lam, eta), eta.expand(3, 4, 4))


# --- Coulomb field -------------------------------------------------------------------------------


def test_single_charge_field_is_inverse_square():
    points = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 4.0]])
    E = cf_eq.electric_field(points, torch.tensor([[0.0, 0.0]]), torch.tensor([2.5]), k=1.0)
    r = points.norm(dim=-1)
    assert torch.allclose(E.norm(dim=-1), 2.5 / r**2)
    assert torch.allclose(E / E.norm(dim=-1, keepdim=True), points / r[:, None])  # radially outward


def test_dipole_field_on_bisector():
    d, q = 0.5, 1.5
    charge_pos = torch.tensor([[d, 0.0], [-d, 0.0]], dtype=torch.float64)
    y = torch.linspace(-3, 3, 13, dtype=torch.float64)
    points = torch.stack([torch.zeros_like(y), y], dim=-1)
    E = cf_eq.electric_field(points, charge_pos, torch.tensor([q, -q], dtype=torch.float64))
    expected = -2 * q * d / (d**2 + y**2) ** 1.5  # from + (at x = +d) towards − (at x = −d)
    assert torch.allclose(E[:, 0], expected)
    assert torch.allclose(E[:, 1], torch.zeros_like(y), atol=1e-12)


# --- RC circuit ----------------------------------------------------------------------------------


def test_capacitor_reaches_63_percent_after_one_time_constant():
    R = rc_t.resistances(0.5, 4.0, 4, CPU)
    C, Vs, dt = 2.0, 5.0, 0.001
    tau = R * C
    result = rc_solve.solve(rc_t.initial_voltage(4, CPU), R, C, Vs, "step", 10.0, t_max=tau.max().item(), dt=dt)
    steps = torch.round(tau / dt).long()
    V_at_tau = result["V"][steps, torch.arange(4)]
    assert torch.allclose(V_at_tau, torch.full((4,), (1 - math.exp(-1)) * Vs), rtol=1e-4)


def test_rc_energy_balance():
    R = rc_t.resistances(1.0, 3.0, 3, CPU).double()
    C, Vs = 1.5, 4.0
    V0 = rc_t.initial_voltage(3, CPU).double()
    result = rc_solve.solve(V0, R, C, Vs, "step", 10.0, t_max=12 * 3.0 * C, dt=0.002)
    U = rc_eq.stored_energy(result["V"][-1], C)
    assert torch.allclose(U, torch.full((3,), 0.5 * C * Vs**2, dtype=U.dtype), rtol=1e-4)  # full: ½CV²
    current, t = result["I"], result["t"].double()
    supplied = torch.trapezoid(Vs * current, t, dim=0)
    heat = torch.trapezoid(current**2 * R, t, dim=0)
    assert torch.allclose(supplied, torch.full((3,), C * Vs**2, dtype=U.dtype), rtol=1e-3)
    assert torch.allclose(heat, U, rtol=1e-3)  # half is lost as heat, whatever R is


# --- Biot–Savart ---------------------------------------------------------------------------------


def test_loop_field_on_axis_matches_formula():
    eps = bs_t.levi_civita(CPU).double()
    R, current = 0.4, 3.0
    mid, dl = bs_t.circle(R, 0.0, 400, +1, CPU)
    z = torch.linspace(-1, 1, 21, dtype=torch.float64)
    axis = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)
    B = bs_eq.biot_savart(axis, mid.double(), dl.double(), eps, current)
    exact = bs_eq.MU0 * current * R**2 / (2 * (R**2 + z**2) ** 1.5)
    assert torch.allclose(B[:, 2], exact, rtol=1e-4)
    assert B[:, :2].abs().max() < 1e-6 * exact.max()  # along the axis by symmetry
    assert torch.allclose(bs_eq.loop_axis_field(z, R, [0.0], [current]), exact)


def test_long_straight_wire():
    eps = bs_t.levi_civita(CPU).double()
    mid, dl = bs_t.straight_wire(200.0, 20000, CPU)
    r = torch.tensor([0.1, 0.3, 1.0], dtype=torch.float64)
    points = torch.stack([r, torch.zeros_like(r), torch.zeros_like(r)], dim=-1)  # on the x axis, wire along z
    B = bs_eq.biot_savart(points, mid.double(), dl.double(), eps, 5.0)
    assert torch.allclose(B[:, 1], bs_eq.wire_field(r, 5.0), rtol=1e-3)  # circles the wire: +y at +x
    assert B[:, [0, 2]].abs().max() < 1e-6 * B.abs().max()


# --- FDTD ----------------------------------------------------------------------------------------


def run_1d(n2: float, n_steps: int, nx: int = 600):
    """A 1D run (Ny = 1): Gaussian pulse from x = 150, interface to refractive index n2 at x = 350."""
    eps_r = torch.ones(nx, 1)
    eps_r[350:] = n2**2
    mask = torch.zeros(nx, 1)
    mask[150] = 1.0
    dt = fd_eq.courant_time_step(1.0, 0.5, dim=1)
    E, H = fd_t.empty_fields(nx, 1, CPU)
    damping = fd_t.damping(nx, 1, 40, 0.1, CPU)
    pulse = fd_solve.gaussian_pulse(30.0, 8.0)
    times, Ez = fd_solve.simulate(E, H, eps_r, damping, mask, pulse, fd_t.levi_civita(CPU), dt, 1.0, n_steps)
    return times, Ez[:, :, 0]


def test_courant_condition_enforced():
    assert fd_eq.courant_time_step(1.0, 1.0, dim=2) == 1 / math.sqrt(2)
    assert fd_eq.courant_time_step(0.5, 0.7, dim=2) * math.sqrt(2) / 0.5 <= 1.0
    for bad in [1.01, 2.0, 0.0]:
        try:
            fd_eq.courant_time_step(1.0, bad, dim=2)
        except ValueError:
            continue
        raise AssertionError(f"Courant number {bad} was accepted")


def test_curl_matches_hand_written_finite_differences():
    eps = fd_t.levi_civita(CPU)
    F = torch.randn(3, 6, 5)
    curl = fd_eq.curl(eps, fd_eq.forward_diff(F, 1.0))
    assert torch.allclose(curl[0, :, :-1], F[2, :, 1:] - F[2, :, :-1])  # (∇×F)_x = ∂_y F_z  (∂_z = 0)
    assert torch.allclose(curl[1, :-1, :], -(F[2, 1:, :] - F[2, :-1, :]))  # (∇×F)_y = −∂_x F_z
    assert torch.allclose(curl[2, :-1, :-1], (F[1, 1:, :-1] - F[1, :-1, :-1]) - (F[0, :-1, 1:] - F[0, :-1, :-1]))


def test_wave_speed_in_vacuum_is_c():
    times, Ez = run_1d(1.0, 700)
    right = Ez[:, 160:]  # only the pulse travelling to the right of the source
    k1, k2 = 300, 600
    speed = (right[k2].argmax() - right[k1].argmax()).item() / (times[k2] - times[k1]).item()
    assert abs(speed - 1.0) < 0.03


def test_reflection_coefficient_at_dielectric_interface():
    n1, n2 = 1.0, 2.0
    _, Ez = run_1d(n2, 2000)
    probe = Ez[:, 250]  # between the source and the interface
    incident = probe.max().item()
    reflected = probe.min().item()  # (n1 − n2)/(n1 + n2) < 0: the echo comes back upside down
    assert abs(reflected / incident - (n1 - n2) / (n1 + n2)) < 0.02
    _, Ez_vacuum = run_1d(1.0, 2000)
    assert Ez_vacuum[:, 250].min().item() > -0.01 * incident  # no interface, no echo: the border absorbs


def test_fdtd_stays_bounded_in_2d():
    nx, ny = 60, 50
    E, H = fd_t.empty_fields(nx, ny, CPU)
    eps_r = fd_t.permittivity(nx, ny, 4.0, CPU)
    damping = fd_t.damping(nx, ny, 10, 0.1, CPU)
    mask = fd_solve.point_source(nx, ny, CPU)
    dt = fd_eq.courant_time_step(1.0, 0.99, dim=2)
    wave = fd_solve.continuous_wave(10.0)
    _, Ez = fd_solve.simulate(E, H, eps_r, damping, mask, wave, fd_t.levi_civita(CPU), dt, 1.0, 400)
    assert torch.isfinite(Ez).all() and Ez.abs().max() < 10


# --- Relativistic particle -----------------------------------------------------------------------


def relativistic_setup(config: str, n: int, strength: float = 1.0):
    eta = rp_t.metric()
    E, B = rp_t.fields(config, strength)
    F_mixed = rp_eq.lower_second(rp_eq.field_tensor(E, B, rp_t.levi_civita()), eta)
    qm, y0 = rp_t.particles(config, n, 0.95)
    return eta, F_mixed, qm, y0


def test_four_velocity_norm_is_preserved():
    for config in ["cyclotron", "accelerate"]:
        eta, F_mixed, qm, y0 = relativistic_setup(config, 3)
        _, traj = rp_solve.solve(y0, qm, F_mixed, t_max=5.0)
        U = traj[:, :, 1]  # (T, P, 4)
        assert torch.allclose(rp_eq.minkowski_norm(y0[:, 1], eta), torch.ones(3, dtype=rp_t.DTYPE))
        assert ((rp_eq.minkowski_norm(U, eta) - 1) / U[..., 0] ** 2).abs().max() < 1e-8


def test_four_force_is_the_lorentz_force():
    eta = rp_t.metric()
    E = torch.tensor([0.3, -0.2, 0.5], dtype=rp_t.DTYPE)
    B = torch.tensor([0.1, 0.4, -0.7], dtype=rp_t.DTYPE)
    F_mixed = rp_eq.lower_second(rp_eq.field_tensor(E, B, rp_t.levi_civita()), eta)
    v = torch.tensor([0.2, 0.5, -0.1], dtype=rp_t.DTYPE)
    gamma = 1 / torch.sqrt(1 - v @ v)
    U = gamma * torch.cat([torch.ones(1, dtype=rp_t.DTYPE), v])[None]  # (1, 4)
    dU = rp_eq.four_force(U, torch.ones(1, dtype=rp_t.DTYPE), F_mixed)[0]
    assert torch.allclose(dU[1:], gamma * (E + torch.linalg.cross(v, B)))
    assert torch.allclose(dU[0], gamma * (E @ v))


def test_relativistic_cyclotron_frequency():
    _, F_mixed, qm, y0 = relativistic_setup("cyclotron", 4, strength=1.5)
    gamma = y0[:, 1, 0]
    t = torch.linspace(0, 30, 3001, dtype=rp_t.DTYPE).numpy()
    _, traj = rp_solve.solve(y0, qm, F_mixed, t_max=30.0, dtau=0.005)
    lab = rp_solve.to_lab_time(traj, t)
    omega = torch.tensor(2 * math.pi * lab["turns"][-1] / t[-1])  # average angular frequency in lab time
    assert torch.allclose(omega, 1.5 / gamma, rtol=1e-4)  # qB/(γm), not qB/m
    v = y0[:, 1, 2] / gamma
    radius = torch.tensor(0.5 * (lab["x"][:, :, 0].max(0) - lab["x"][:, :, 0].min(0)))
    assert torch.allclose(radius, gamma * v / 1.5, rtol=1e-3)  # r = γmv/(qB)


def test_constant_field_gives_hyperbolic_motion():
    _, F_mixed, qm, y0 = relativistic_setup("accelerate", 3, strength=0.8)
    t = torch.linspace(0, 10, 101, dtype=rp_t.DTYPE)
    _, traj = rp_solve.solve(y0, qm, F_mixed, t_max=10.0)
    x = torch.tensor(rp_solve.to_lab_time(traj, t.numpy())["x"][:, :, 0])  # (F, P)
    a = qm * 0.8  # (P,)
    exact = (torch.sqrt(1 + (t[:, None] * a) ** 2) - 1) / a
    assert torch.allclose(x, exact, atol=1e-3)
    assert (x <= t[:, None] + 1e-9).all()  # never faster than light


# --- Waveguide modes -----------------------------------------------------------------------------


def test_waveguide_cutoffs_match_exact():
    a, b = 2.286, 1.016  # WR-90
    found = wg_solve.solve(a, b, 24, 8)
    for mode in found:
        assert abs(mode["k_c"] - wg_eq.exact_cutoff(*mode["mn"], a, b)) / mode["k_c"] < 0.01
    assert [(m["kind"], m["mn"]) for m in found[:3]] == [("TE", (1, 0)), ("TE", (2, 0)), ("TE", (0, 1))]
    assert abs(wg_solve.cutoff_frequency_GHz(found[0]["k_c"]) - 6.56) < 0.02


def test_dirichlet_cutoffs_and_orthonormal_modes():
    a, b = 2.0, 1.0
    nx, ny = wg_t.grid_size(a, b, 20)
    _, _, hx, hy = wg_t.grid(a, b, nx, ny, "TM")
    k_c, psi = wg_eq.modes(wg_eq.laplacian(nx, ny, hx, hy, "dirichlet"), 5, skip_zero=False)
    exact = sorted(wg_eq.exact_cutoff(m, n, a, b) for m in range(1, 6) for n in range(1, 4))[:5]
    assert torch.allclose(k_c, torch.tensor(exact, dtype=torch.float64), rtol=0.01)
    gram = torch.einsum("nk,nl->kl", psi, psi)  # ψ_k · ψ_l, summed over grid points n
    assert torch.allclose(gram, torch.eye(5, dtype=torch.float64), atol=1e-10)
