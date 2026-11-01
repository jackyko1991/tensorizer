import math

import pytest
import torch

from problems.fluids.bernoulli_venturi import equation as bv_eq
from problems.fluids.bernoulli_venturi import solve as bv_solve
from problems.fluids.bernoulli_venturi import tensors as bv_t
from problems.fluids.burgers_shock import equation as bu_eq
from problems.fluids.burgers_shock import solve as bu_solve
from problems.fluids.burgers_shock import tensors as bu_t
from problems.fluids.cylinder_drag import equation as cd_eq
from problems.fluids.cylinder_drag import solve as cd_solve
from problems.fluids.cylinder_drag import tensors as cd_t
from problems.fluids.elastic_waves import equation as ew_eq
from problems.fluids.elastic_waves import solve as ew_solve
from problems.fluids.elastic_waves import tensors as ew_t
from problems.fluids.hydrostatics_buoyancy import equation as hb_eq
from problems.fluids.hydrostatics_buoyancy import solve as hb_solve
from problems.fluids.hydrostatics_buoyancy import tensors as hb_t
from problems.fluids.lid_driven_cavity import equation as ld_eq
from problems.fluids.lid_driven_cavity import solve as ld_solve
from problems.fluids.lid_driven_cavity import tensors as ld_t
from problems.fluids.navier_stokes_pipe_flow import equation as ns_eq
from problems.fluids.navier_stokes_pipe_flow import solve as ns_solve
from problems.fluids.navier_stokes_pipe_flow import tensors as ns_t
from problems.fluids.stress_strain import equation as ss_eq
from problems.fluids.stress_strain import solve as ss_solve
from problems.fluids.stress_strain import tensors as ss_t
from problems.fluids.torricelli_draining import equation as td_eq
from problems.fluids.torricelli_draining import solve as td_solve
from problems.fluids.torricelli_draining import tensors as td_t

CPU = torch.device("cpu")
G = 9.81


# ---------------------------------------------------------------- hydrostatics and buoyancy


def test_hydrostatic_pressure_is_linear_in_depth():
    g = hb_t.gravity(G, CPU)
    points = hb_t.tank_points(1.0, 2.0, 5, 41, CPU)  # (nz, nx, 2), z from -2 to 0
    p = hb_eq.pressure(points, g, 101_325.0, 1000.0)
    depth = -points[..., 1]
    assert torch.allclose(p, 101_325.0 + 1000.0 * G * depth, rtol=1e-6)
    assert torch.allclose(p[:, 0], p[:, -1])  # same depth ⇒ same pressure


def test_floating_fraction_matches_archimedes():
    rho_b = hb_t.densities(5, 200.0, 900.0, CPU)
    y0 = hb_t.initial_state(5, 0.1, CPU)
    _, traj = hb_solve.solve(y0, rho_b, 0.2, depth=1.0, t_max=8.0)
    assert torch.allclose(hb_solve.floating_fraction(traj, 0.2), rho_b / 1000.0, atol=2e-3)


def test_dense_blocks_sink_to_the_floor():
    rho_b = torch.tensor([1200.0, 2500.0])
    _, traj = hb_solve.solve(hb_t.initial_state(2, 0.1, CPU), rho_b, 0.2, depth=1.0, t_max=5.0)
    assert torch.allclose(traj[-1, :, 0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(traj[-1, :, 1], torch.zeros(2))


# ---------------------------------------------------------------- Bernoulli / Venturi


def test_venturi_continuity_and_bernoulli():
    r1, ratio, u1, p1, rho = 0.05, 4.0, 1.5, 20_000.0, 1000.0
    x = bv_t.axis(500, CPU)
    flow = bv_solve.steady_flow(x, r1, r1 / math.sqrt(ratio), u1, p1, rho)
    Q = flow["A"] * flow["u"]
    assert torch.allclose(Q, torch.full_like(Q, flow["Q"]), rtol=1e-5)  # same flow rate everywhere
    u2 = flow["u"].max().item()
    assert u2 / u1 == pytest.approx(ratio, rel=1e-4)  # v2/v1 = A1/A2
    dp = p1 - flow["p"].min().item()
    assert dp == pytest.approx(0.5 * rho * (u2**2 - u1**2), rel=1e-4)
    energy = flow["p"] + 0.5 * rho * flow["u"] ** 2
    assert torch.allclose(energy, energy[0].expand_as(energy), rtol=1e-6)


def test_venturi_meter_recovers_flow_rate():
    A1, A2, rho, Q = 0.01, 0.0025, 1000.0, 0.02
    u1, u2 = Q / A1, Q / A2
    dp = 0.5 * rho * (u2**2 - u1**2)
    assert bv_eq.venturi_flow_rate(dp, A1, A2, rho) == pytest.approx(Q, rel=1e-9)


def test_tracers_move_faster_in_the_throat():
    r1, r2, Q = 0.05, 0.025, 0.004
    x0 = torch.tensor([0.05, 0.45])  # inlet and throat
    times, traj = bv_solve.move_tracers(x0, r1, r2, Q, t_max=0.01, dt=0.001)
    speed = (traj[1] - traj[0]) / (times[1] - times[0])
    assert (speed[1] / speed[0]).item() == pytest.approx((r1 / r2) ** 2, rel=1e-2)


# ---------------------------------------------------------------- Torricelli


def test_torricelli_matches_analytic_solution_and_drain_time():
    d = td_t.hole_diameters(3, 0.01, 0.03, CPU)
    a, A, h0 = td_t.circle_area(d), td_t.circle_area(0.4), 1.2
    T = td_eq.drain_time(a, A, h0, G)
    times, levels = td_solve.solve(torch.full((3,), h0), a, A, G, t_max=1.1 * T.max().item())
    exact = td_eq.analytic_height(times, a, A, h0, G)
    assert (levels - exact).abs().max() < 1e-5
    assert torch.allclose(td_solve.numerical_drain_time(times, levels), T, rtol=3e-3)


def test_drain_time_scaling():
    a = torch.tensor([1e-4, 4e-4])
    T = td_eq.drain_time(a, 0.1, 1.0, G)
    assert (T[0] / T[1]).item() == pytest.approx(4.0)  # 4× the hole area → 4× faster
    assert (td_eq.drain_time(a, 0.1, 4.0, G) / T).tolist() == pytest.approx([2.0, 2.0])  # 4× height → 2× time


# ---------------------------------------------------------------- Burgers


@pytest.fixture(scope="module")
def burgers_run():
    nu, n = 0.05, 256
    x, dx = bu_t.grid(n, CPU)
    D, L = bu_t.derivative_matrices(n, dx, CPU)
    u0 = bu_t.initial_wave(x, 1.0, 0.0)
    dt = bu_solve.time_step(dx, 1.0, nu)
    times, u = bu_solve.simulate(u0, D, L, nu, dt, 2.0)
    return {"x": x, "dx": dx, "D": D, "u": u, "times": times, "nu": nu}


def test_burgers_conserves_mass(burgers_run):
    m = bu_eq.mass(burgers_run["u"], burgers_run["dx"])
    assert (m - m[0]).abs().max() < 1e-10


def test_burgers_matches_cole_hopf(burgers_run):
    r = burgers_run
    for t in (0.5, 1.0, 2.0):
        k = int((r["times"] - t).abs().argmin())
        exact = bu_solve.cole_hopf(r["x"], r["times"][k].item(), r["nu"], 1.0, 0.0)
        assert (r["u"][k] - exact).abs().max() < 0.02


def test_burgers_shock_forms_near_breaking_time():
    nu, n = 0.01, 512
    x, dx = bu_t.grid(n, CPU)
    D, L = bu_t.derivative_matrices(n, dx, CPU)
    times, u = bu_solve.simulate(bu_t.initial_wave(x, 1.0, 0.0), D, L, nu, bu_solve.time_step(dx, 1.0, nu), 2.0)
    steep = (-bu_eq.slope(u, D)).max(dim=-1).values
    t_half = times[int((steep > 0.5 * steep.max()).int().argmax())].item()
    assert 0.9 < t_half < 1.3  # t* = 1/max(-u0') = 1
    k = int((times - 0.5).abs().argmin())
    assert steep[k].item() == pytest.approx(1 / (1 - times[k].item()), rel=0.05)  # inviscid slope 1/(1 - t) early on


def test_cole_hopf_initial_condition():
    x, _ = bu_t.grid(64, CPU)
    early = bu_solve.cole_hopf(x, 1e-4, 0.05, 1.0, 0.3, n_y=40000)
    assert torch.allclose(early, 0.3 + torch.sin(x), atol=1e-2)


# ---------------------------------------------------------------- Navier–Stokes pipe flow


@pytest.fixture(scope="module")
def pipe_run():
    R, nu, G_p, rho = 0.005, 1e-6, 0.5, 1000.0
    r, dr = ns_t.grid(40, R, CPU)
    L = ns_t.radial_laplacian(r, dr)
    times, u = ns_solve.simulate(ns_t.at_rest(40, CPU), L, G_p, rho, nu, dr, 2.0 * R**2 / nu)
    return {"r": r, "dr": dr, "L": L, "times": times, "u": u, "R": R, "nu": nu, "G": G_p, "mu": rho * nu, "rho": rho}


def test_pipe_laplacian_is_exact_for_the_parabola(pipe_run):
    r, L = pipe_run["r"], pipe_run["L"]
    u = (pipe_run["R"] ** 2 - r**2)[:-1]  # unknowns: everything but the wall
    assert torch.allclose(torch.einsum("ij,j->i", L, u), torch.full_like(u, -4.0), rtol=1e-9)  # ∇²(R² - r²) = -4


def test_pipe_reaches_hagen_poiseuille(pipe_run):
    p = pipe_run
    u_s = ns_eq.hagen_poiseuille(p["r"], p["G"], p["mu"], p["R"])
    assert torch.allclose(p["u"][-1], u_s, rtol=1e-4, atol=1e-9)
    Q = ns_eq.flow_rate(p["u"][-1], p["r"]).item()
    assert Q == pytest.approx(math.pi * p["G"] * p["R"] ** 4 / (8 * p["mu"]), rel=2e-3)
    tau_w = ns_eq.wall_shear_stress(p["u"][-1], p["dr"], p["mu"]).item()
    assert tau_w == pytest.approx(p["G"] * p["R"] / 2, rel=1e-3)  # force balance G πR² = τ_w 2πR


def test_pipe_startup_matches_bessel_series(pipe_run):
    p = pipe_run
    tau = p["R"] ** 2 / p["nu"]
    for frac in (0.05, 0.2):
        k = int((p["times"] - frac * tau).abs().argmin())
        exact = ns_solve.series_solution(p["r"], p["times"][k].item(), p["G"], p["mu"], p["rho"], p["R"])
        assert (p["u"][k] - exact).abs().max() < 2e-3 * exact.max()


def test_pipe_time_scale_is_R2_over_nu(pipe_run):
    p = pipe_run
    tau = p["R"] ** 2 / p["nu"]
    centre = p["u"][:, 0] / p["u"][-1, 0]
    t63 = p["times"][int((centre > 0.63).int().argmax())].item() / tau
    assert 0.1 < t63 < 0.3  # slowest mode e^{-λ₁² ν t/R²}, λ₁ = 2.405 ⇒ about 0.19 R²/ν


def test_bessel_zeros():
    lam = ns_solve.bessel_zeros(3)
    assert lam.tolist() == pytest.approx([2.404826, 5.520078, 8.653728], rel=1e-6)


# ---------------------------------------------------------------- stress and strain


def material():
    lam, mu = ss_eq.lame(200e9, 0.3)
    delta = ss_t.kronecker(CPU)
    return lam, mu, delta, ss_eq.stiffness(lam, mu, delta)


def test_stiffness_tensor_symmetries():
    _, _, _, C = material()
    assert torch.allclose(C, C.transpose(0, 1))  # minor symmetry ij
    assert torch.allclose(C, C.transpose(2, 3))  # minor symmetry kl
    assert torch.allclose(C, C.permute(2, 3, 0, 1))  # major symmetry


def test_einsum_hooke_matches_lame_form():
    lam, mu, delta, C = material()
    a = torch.randn(5, 3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 1e-3
    eps = 0.5 * (a + a.transpose(-1, -2))
    assert torch.allclose(ss_eq.stress(C, eps), ss_eq.lame_stress(eps, lam, mu, delta), rtol=1e-10)


def test_uniaxial_stress_gives_young_and_poisson():
    E, nu = 200e9, 0.3
    _, _, _, C = material()
    e = 1e-3
    eps = torch.diag(torch.tensor([e, -nu * e, -nu * e], dtype=torch.float64))
    sigma = ss_eq.stress(C, eps)
    expected = torch.zeros(3, 3, dtype=torch.float64)
    expected[0, 0] = E * e
    assert torch.allclose(sigma, expected, atol=1e-3 * E * e)
    S = ss_solve.compliance(C)
    back = torch.einsum("ijkl,kl->ij", S, sigma)
    assert torch.allclose(back, eps, atol=1e-12)


def test_principal_stresses_invariant_under_rotation():
    _, _, _, C = material()
    sigma = torch.tensor([[100.0, 60.0, 0.0], [60.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    a = 0.7
    R = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], dtype=torch.float64)
    rotated = torch.einsum("ia,ab,jb->ij", R, sigma, R)
    v1, _ = ss_eq.principal_stresses(sigma)
    v2, _ = ss_eq.principal_stresses(rotated)
    assert torch.allclose(v1, v2, atol=1e-9)
    centre, radius = ss_eq.mohr_circle(sigma)
    assert (centre + radius).item() == pytest.approx(v1.max().item())  # Mohr's circle hits σ1
    assert torch.allclose(ss_eq.mohr_circle(rotated)[1], radius)


def test_strain_from_displacement_gradient_round_trip():
    _, _, _, C = material()
    x = ss_t.block_points(7, CPU)
    loads = ss_t.loading(5, 100e6, 60e6, CPU)
    result = ss_solve.load_path(loads, C, x)
    assert torch.allclose(result["sigma"], loads, atol=1e-3 * 100e6)  # σ = C : ε(∇u) recovers the applied stress


# ---------------------------------------------------------------- lid-driven cavity


@pytest.fixture(scope="module")
def cavity_run():
    n = 41
    y, h = ld_t.grid(n, CPU)
    V, lam = ld_t.laplacian_eigen(n - 2, h, CPU)
    omega, _ = ld_t.at_rest(n, CPU)
    times, omegas, psis = ld_solve.simulate(omega, V, lam, h, 0.01, 1.0, 25.0, save_every=500)
    return {"y": y, "h": h, "V": V, "lam": lam, "omega": omegas[-1], "psi": psis[-1]}


def test_cavity_poisson_solver_residual(cavity_run):
    c = cavity_run
    f = torch.randn(39, 39, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    psi = ld_solve.poisson_solve(f, c["V"], c["lam"])
    assert (ld_eq.laplacian(psi, c["h"]) - f).abs().max() < 1e-9 * f.abs().max()
    assert torch.allclose(ld_eq.laplacian(c["psi"], c["h"]), -c["omega"][1:-1, 1:-1], atol=1e-8)


def test_cavity_velocity_is_divergence_free(cavity_run):
    u = ld_eq.velocity(cavity_run["psi"], cavity_run["h"])
    assert ld_eq.divergence(u, cavity_run["h"]).abs().max() < 1e-10


def test_cavity_matches_ghia_at_re100(cavity_run):
    u = ld_solve.centreline_u(cavity_run["psi"], cavity_run["h"])
    assert ld_solve.ghia_deviation(cavity_run["y"], u) < 0.02  # |u| ≤ 1, Ghia min is -0.21
    psi = cavity_run["psi"]
    i, j = divmod(int(psi.argmin()), psi.shape[0])
    assert abs(cavity_run["y"][i] - 0.617) < 0.05 and abs(cavity_run["y"][j] - 0.734) < 0.05
    assert psi.min().item() == pytest.approx(-0.1034, rel=0.05)


# ---------------------------------------------------------------- cylinder drag (lattice Boltzmann)


def test_lbm_equilibrium_moments():
    e, w, _ = cd_t.lattice(CPU)
    gen = torch.Generator().manual_seed(2)
    rho = 1 + 0.1 * torch.rand(6, 5, generator=gen)
    u = 0.1 * torch.randn(6, 5, 2, generator=gen)
    f_eq = cd_eq.equilibrium(rho, u, e, w)
    assert torch.allclose(cd_eq.density(f_eq), rho, atol=1e-6)
    assert torch.allclose(cd_eq.momentum(f_eq, e), rho[..., None] * u, atol=1e-6)
    assert w.sum().item() == pytest.approx(1.0)


def test_lbm_conserves_mass_in_a_closed_box():
    nx, ny = 40, 30
    e, w, opp = cd_t.lattice(CPU)
    solid = cd_t.cylinder(nx, ny, 15, 15, 4, CPU)
    links = cd_t.boundary_links(solid, e)
    f = cd_t.initial_populations(nx, ny, 0.08, e, w, CPU)
    mass0 = f.sum().item()
    result = cd_solve.simulate(f, solid, links, e, w, opp, 0.6, 0.08, 300, 100, open_ends=False)
    assert result["f"].sum().item() == pytest.approx(mass0, rel=1e-5)


@pytest.fixture(scope="module")
def cylinder_run():
    nx, ny, D, U = 180, 56, 10, 0.12
    e, w, opp = cd_t.lattice(CPU)
    solid = cd_t.cylinder(nx, ny, 40.0, ny / 2, D / 2, CPU)
    links = cd_t.boundary_links(solid, e)
    f = cd_t.initial_populations(nx, ny, U, e, w, CPU)
    tau = cd_eq.viscosity_to_tau(U * D / 100.0)
    result = cd_solve.simulate(f, solid, links, e, w, opp, tau, U, 5000, 1000)
    return cd_solve.coefficients(result["force"], U, D), U, D


def test_cylinder_drag_positive_and_right_order(cylinder_run):
    coeffs, _, _ = cylinder_run
    cd = coeffs[2500:, 0]
    assert (cd > 0).all()
    assert 1.0 < cd.mean().item() < 3.0


def test_cylinder_lift_oscillates_with_zero_mean_and_sheds_periodically(cylinder_run):
    coeffs, U, D = cylinder_run
    cl = coeffs[2500:, 1]
    assert cl.abs().max().item() > 0.1  # the wake really oscillates
    assert abs(cl.mean().item()) < 0.1 * cl.abs().max().item()
    assert 0.12 < cd_solve.strouhal(coeffs[:, 1], U, D) < 0.3


# ---------------------------------------------------------------- elastic waves


def test_elastic_wave_speeds_from_front_positions():
    n, lam, mu, rho = 161, 2.0, 1.0, 1.0
    c_p, c_s = ew_eq.wave_speeds(lam, mu, rho)
    dt = ew_eq.stable_time_step(1.0, c_p, c_s, 0.6)
    u, u_prev = ew_t.at_rest(n, CPU)
    force = ew_t.point_force(n, 0.0, 1.5, CPU)
    result = ew_solve.simulate(u, u_prev, force, ew_t.sponge(n, 20, 0.08, CPU), 1.0, dt, lam, mu, rho, 0.08, 40.0)
    times = result["times"]
    a, b = int((times - 25).abs().argmin()), int((times - 38).abs().argmin())
    span = (times[b] - times[a]).item()
    cp_measured = (ew_solve.front_radius(result["div"][b], "x") - ew_solve.front_radius(result["div"][a], "x")) / span
    cs_measured = (ew_solve.front_radius(result["curl"][b], "y") - ew_solve.front_radius(result["curl"][a], "y")) / span
    assert cp_measured == pytest.approx(c_p, rel=0.1)
    assert cs_measured == pytest.approx(c_s, rel=0.1)


def test_elastic_cfl_is_enforced():
    with pytest.raises(ValueError):
        ew_eq.stable_time_step(1.0, 2.0, 1.0, 1.2)
    assert ew_eq.stable_time_step(1.0, 2.0, 1.0, 1.0) == pytest.approx(1 / math.sqrt(5))


def test_navier_cauchy_einsums_on_a_plane_wave():
    # u = (sin kx, 0): ∂_i∂_j u_j = (-k² sin kx, 0) and ∂_j∂_j u_i = (-k² sin kx, 0) ⇒ ü = -(λ+2μ)k² u/ρ (a P wave)
    n = 64
    k = 2 * math.pi / n
    x = torch.arange(n, dtype=torch.float32)
    u = torch.zeros(2, n, n)
    u[0] = torch.sin(k * x)[:, None]
    acc = ew_eq.acceleration(u, 1.0, 2.0, 1.0, 1.0)
    k_eff2 = 2 - 2 * math.cos(k)  # second difference of sin is -k_eff² sin
    assert torch.allclose(acc[0], -4.0 * k_eff2 * u[0], atol=1e-5)
    assert acc[1].abs().max() < 1e-5
