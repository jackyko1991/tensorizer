import math

import pytest
import torch

from problems.waves.beats_superposition import equation as bs_eq
from problems.waves.beats_superposition import solve as bs_solve
from problems.waves.beats_superposition import tensors as bs_t
from problems.waves.dispersive_waves import equation as dw_eq
from problems.waves.dispersive_waves import solve as dw_solve
from problems.waves.dispersive_waves import tensors as dw_t
from problems.waves.doppler_effect import equation as de_eq
from problems.waves.doppler_effect import solve as de_solve
from problems.waves.doppler_effect import tensors as de_t
from problems.waves.kdv_solitons import equation as kdv_eq
from problems.waves.kdv_solitons import solve as kdv_solve
from problems.waves.kdv_solitons import tensors as kdv_t
from problems.waves.membrane_modes import equation as mm_eq
from problems.waves.membrane_modes import solve as mm_solve
from problems.waves.membrane_modes import tensors as mm_t
from problems.waves.nonlinear_schrodinger import equation as nls_eq
from problems.waves.nonlinear_schrodinger import solve as nls_solve
from problems.waves.nonlinear_schrodinger import tensors as nls_t
from problems.waves.ripple_tank import equation as rt_eq
from problems.waves.ripple_tank import solve as rt_solve
from problems.waves.ripple_tank import tensors as rt_t
from problems.waves.standing_waves import equation as sw_eq
from problems.waves.standing_waves import solve as sw_solve
from problems.waves.standing_waves import tensors as sw_t
from problems.waves.string_junction import equation as sj_eq
from problems.waves.string_junction import solve as sj_solve
from problems.waves.string_junction import tensors as sj_t

CPU = torch.device("cpu")


# --- standing waves -------------------------------------------------------------------------------------------------


def test_standing_wave_frequencies_are_harmonics():
    L, c = 2.0, 1.5
    x, dx = sw_t.grid(L, 201, CPU)
    D = sw_t.stencil(len(x) - 2, dx, CPU).double()
    f = sw_eq.mode_frequencies(D, c)[:5]
    expected = torch.arange(1, 6, dtype=f.dtype) * c / (2 * L)
    assert torch.allclose(f, expected, rtol=1e-3)


def test_standing_wave_harmonic_returns_after_one_period_and_conserves_energy():
    L, c = 1.0, 1.0
    x, dx = sw_t.grid(L, 101, CPU)
    D = sw_t.stencil(len(x) - 2, dx, CPU)
    u0 = sw_t.initial_shapes(x, L, n_harmonics=2, pluck_at=0.3)  # rows: n = 1, n = 2, plucked
    times, traj, dt = sw_solve.solve(u0, D, c, dx, t_max=2 * L / c)  # one fundamental period
    assert torch.allclose(traj[-1, :2], u0[:2], atol=2e-2)  # harmonics are back
    # half a period of n = 1: the shape is upside down
    half = int(torch.argmin((times - L / c).abs()))
    assert torch.allclose(traj[half, 0], -u0[0], atol=2e-2)
    E = sw_eq.energy(traj, dx, dt, c)  # (T-2, B)
    assert torch.allclose(E, E[0].expand_as(E), rtol=1e-2)
    # analytic energy of sin(nπx/L) at rest: ½ c² (nπ/L)² · L/2
    n = torch.tensor([1.0, 2.0])
    assert torch.allclose(E[0, :2], 0.5 * c**2 * (n * math.pi / L) ** 2 * L / 2, rtol=2e-2)


# --- beats --------------------------------------------------------------------------------------------------------


def test_beat_frequency_is_frequency_difference():
    omega = bs_t.angular_frequencies(1.0, 1.25, 2, CPU)
    k = bs_t.wavenumbers(omega, "none")
    A = bs_t.amplitudes(2, CPU)
    x, t = bs_t.grids(10.0, 40.0, CPU)
    res = bs_solve.solve(A, k, omega, x, t)
    dt = (t[1] - t[0]).item()
    assert bs_solve.beat_frequency(res["trace"], dt, f_min=1.0) == pytest.approx(0.25, abs=0.01)
    # envelope period 1/|f1 - f2| = 4 s: the envelope at x = 0 is |cos(π Δf t)|, back to 1 after 4 s
    env = res["trace_env"]
    expected = torch.abs(torch.cos(math.pi * 0.25 * t))
    assert torch.allclose(env, expected, atol=1e-4)
    assert torch.all(res["trace"].abs() <= env + 1e-5)


@pytest.mark.parametrize("dispersion, ratio", [("none", 1.0), ("water", 0.5), ("capillary", 1.5)])
def test_group_over_phase_velocity_equals_dispersion_power(dispersion, ratio):
    omega = bs_t.angular_frequencies(1.0, 1.01, 2, CPU).double()
    k = bs_t.wavenumbers(omega, dispersion)
    assert bs_eq.group_velocity(k, omega) / bs_eq.phase_velocity(k, omega) == pytest.approx(ratio, rel=1e-2)


# --- Doppler ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mach", [0.0, 0.3, 0.8])
def test_doppler_heard_frequencies_match_formula(mach):
    c, f, t_max = de_t.C_SOUND, 2.0, 10.0
    v_s = mach * c
    x_start = -0.5 * v_s * t_max
    t_e = de_t.emission_times(f, t_max, CPU)
    s = de_t.source_position(t_e, v_s, x_start)
    obs = de_t.observers(x_start, v_s, t_max, c, CPU)
    heard = de_eq.heard_frequency(de_solve.solve(t_e, s, obs, c, t_max)["arrivals"])
    ahead, behind = de_eq.doppler_formula(f, c, v_s)
    assert heard[0].item() == pytest.approx(f * c / (c - v_s), rel=1e-4)
    assert heard[1].item() == pytest.approx(f * c / (c + v_s), rel=1e-4)
    assert (ahead, behind) == pytest.approx((heard[0].item(), heard[1].item()), rel=1e-4)


def test_mach_cone_angle():
    c, mach, t_max = de_t.C_SOUND, 1.6, 8.0
    v_s = mach * c
    t_e = de_t.emission_times(1.0, t_max, CPU).double()
    s = de_t.source_position(t_e, v_s, 0.0)
    radii = de_eq.front_radii(t_max, t_e, c)
    mu = de_eq.cone_half_angle(s, radii, de_t.source_position(torch.tensor(t_max, dtype=torch.float64), v_s, 0.0))
    assert torch.allclose(mu, torch.full_like(mu, math.asin(1 / mach)), atol=1e-9)
    # subsonic: the source sits inside all of its fronts, so there is no cone
    s_sub = de_t.source_position(t_e, 0.5 * c, 0.0)
    now = de_t.source_position(torch.tensor(t_max, dtype=torch.float64), 0.5 * c, 0.0)
    assert len(de_eq.cone_half_angle(s_sub, radii, now)) == 0


# --- string junction ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mu_ratio", [4.0, 0.25, 9.0])
def test_string_junction_reflection_transmission_and_energy(mu_ratio):
    T, mu1, mu2 = sj_t.TENSION, sj_t.MU1, mu_ratio
    c1, c2 = sj_t.speed(mu1), sj_t.speed(mu2)
    x, dx = sj_t.grid(1001, CPU)
    mu = sj_t.density(x, mu1, mu2)
    dt = 0.9 * dx / max(c1, c2)
    u_prev, u0 = sj_t.right_moving_pulse(x, 0.04, c1, dt)
    t_meas = sj_solve.measuring_time(c1, c2, sj_t.X0)
    times, traj = sj_solve.solve(u_prev, u0, mu, T, dx, dt, t_meas)
    R_num, T_num = sj_solve.measure(x, traj[-1])
    Z1, Z2 = sj_eq.impedance(T, mu1), sj_eq.impedance(T, mu2)
    R, Tr = sj_eq.reflection_transmission(Z1, Z2)
    assert R_num == pytest.approx(R, abs=0.01)
    assert T_num == pytest.approx(Tr, abs=0.015)
    assert R**2 + Z2 / Z1 * Tr**2 == pytest.approx(1.0)  # exact identity
    assert R_num**2 + Z2 / Z1 * T_num**2 == pytest.approx(1.0, abs=0.02)  # measured flux balance
    E = sj_eq.energy(traj, mu, T, dx, dt)
    assert torch.allclose(E, E[0].expand_as(E), rtol=1e-3)


# --- dispersive waves ---------------------------------------------------------------------------------------------


def test_deep_water_group_velocity_is_half_phase_velocity():
    g, k0 = dw_t.G, 2 * math.pi / 12.0
    x, k, _ = dw_t.grid(2048, CPU)
    a0 = dw_t.wave_packet(x, dw_t.X0, 30.0, k0)  # wide packet: narrow spectrum
    times, a = dw_solve.solve(a0, k, g, t_max=40.0, n_times=401)
    v_g = dw_solve.envelope_speed(times, dw_solve.centroid(x, a))
    v_p = dw_solve.crest_speed(x, a, times)
    assert v_p == pytest.approx(dw_eq.phase_velocity(k0, g), rel=1e-2)
    assert v_g == pytest.approx(dw_eq.group_velocity(k0, g), rel=1e-2)
    assert v_g / v_p == pytest.approx(0.5, rel=2e-2)


def test_dispersive_packet_conserves_energy_and_spreads():
    x, k, _ = dw_t.grid(2048, CPU)
    a0 = dw_t.wave_packet(x, dw_t.X0, 8.0, 2 * math.pi / 10.0)
    _, a = dw_solve.solve(a0, k, dw_t.G, t_max=60.0, n_times=7)
    E = dw_eq.wave_energy(a)
    assert torch.allclose(E, E[0].expand_as(E), rtol=1e-10)
    assert a[-1].abs().max() < 0.8 * a[0].abs().max()  # dispersion lowers and widens the envelope


# --- membrane modes -----------------------------------------------------------------------------------------------


def test_square_membrane_frequencies_and_orthonormal_modes():
    n = 30
    X, Y, h = mm_t.grid(n, CPU)
    D = mm_t.second_difference(n, h, CPU)
    inside = mm_t.drum_mask(X, Y, "square")
    omega, shapes = mm_solve.solve(D, inside, mm_t.C)
    ratios = (omega[:6] / omega[0]).tolist()
    exact = [math.sqrt((m * m + q * q) / 2) for m, q in [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]]
    assert ratios == pytest.approx(exact, rel=5e-3)
    assert omega[0].item() == pytest.approx(math.pi * math.sqrt(2) * mm_t.C / mm_t.SIDE, rel=2e-3)
    lap = mm_eq.restrict(mm_eq.laplacian_2d(D), inside)
    _, phi = mm_eq.modes(lap, mm_t.C)
    assert torch.allclose(mm_eq.overlap(phi), torch.eye(phi.shape[1], dtype=phi.dtype), atol=1e-10)


def test_circular_membrane_fundamental_close_to_bessel_zero():
    n = 40
    X, Y, h = mm_t.grid(n, CPU)
    omega, _ = mm_solve.solve(mm_t.second_difference(n, h, CPU), mm_t.drum_mask(X, Y, "circle"), mm_t.C)
    assert omega[0].item() == pytest.approx(2.40483 * mm_t.C / (mm_t.SIDE / 2), rel=3e-2)  # staircase rim
    assert (omega[1] / omega[0]).item() == pytest.approx(3.83171 / 2.40483, rel=1e-2)


# --- KdV ----------------------------------------------------------------------------------------------------------


def test_kdv_single_soliton_speed_is_twice_its_height():
    c = 2.0
    x, k, dx = kdv_t.grid(512, CPU)
    u0 = kdv_eq.soliton(x, c, 30.0)
    assert kdv_eq.soliton(torch.tensor([30.0], dtype=torch.float64), c, 30.0).item() == pytest.approx(c / 2)
    times, u = kdv_solve.solve(u0, k, kdv_t.dealias_mask(k), t_max=10.0, n_frames=11)
    positions = torch.tensor([kdv_solve.peak_position(x, ui) for ui in u], dtype=torch.float64)
    speed = ((positions[-1] - positions[0]) / (times[-1] - times[0])).item()
    assert speed == pytest.approx(c, rel=5e-3)
    assert u[-1].max().item() == pytest.approx(u0.max().item(), rel=5e-3)  # keeps its height


def test_kdv_collision_conserves_mass_momentum_and_shifts_solitons():
    c1, c2, t_max = 4.0, 1.0, 12.0
    x, k, dx = kdv_t.grid(1024, CPU)
    u0 = kdv_t.two_solitons(x, c1, c2)
    times, u = kdv_solve.solve(u0, k, kdv_t.dealias_mask(k), t_max)
    M, P = kdv_eq.mass(u, dx), kdv_eq.momentum(u, dx)
    assert torch.allclose(M, M[0].expand_as(M), rtol=1e-10)
    assert torch.allclose(P, P[0].expand_as(P), rtol=1e-3)
    x1 = kdv_solve.peak_position(x, u[-1])
    x2 = kdv_solve.second_peak_position(x, u[-1], x1)
    d1, d2 = kdv_eq.phase_shifts(c1, c2)
    assert x1 - (kdv_t.X_TALL + c1 * t_max) == pytest.approx(d1, abs=0.05)  # tall soliton jumps ahead
    assert x2 - (kdv_t.X_SHORT + c2 * t_max) == pytest.approx(-d2, abs=0.05)  # short one falls behind


# --- NLS ----------------------------------------------------------------------------------------------------------


def test_nls_bright_soliton_keeps_its_shape_and_invariants():
    eta, v, dt, t_max = 1.5, nls_t.SOLITON_SPEED, 0.005, 10.0
    x, k, dx = nls_t.grid(512, CPU)
    psi0 = nls_t.initial_wave(x, "soliton", 0.0, eta)
    times, psi = nls_solve.solve(psi0, nls_t.half_linear_step(k, dt), dt, t_max, n_frames=11)
    exact = nls_eq.bright_soliton(x + 10.0, t_max, eta, v)
    assert torch.allclose(psi[-1].abs(), exact.abs(), atol=1e-3)
    N, H = nls_eq.norm(psi, dx), nls_eq.hamiltonian(psi, k, dx)
    assert N[0].item() == pytest.approx(2 * eta, rel=1e-6)  # ∫ η² sech² = 2η
    assert torch.allclose(N, N[0].expand_as(N), rtol=1e-10)
    assert torch.allclose(H, H[0].expand_as(H), rtol=1e-6)


def test_nls_modulational_instability_and_peregrine_peak():
    dt = 0.005
    x, k, dx = nls_t.grid(512, CPU)
    half = nls_t.half_linear_step(k, dt)
    # Peregrine breather started at t = -5 peaks at |psi| = 3 at t = 5
    times, psi = nls_solve.solve(nls_t.initial_wave(x, "peregrine", 0.0, 1.0), half, dt, 7.0, n_frames=71)
    peak = psi.abs().max(dim=1).values
    assert peak.max().item() == pytest.approx(3.0, abs=0.02)
    assert times[peak.argmax()].item() == pytest.approx(5.0, abs=0.15)
    # a noisy plane wave grows peaks well above the background while N and H stay put
    times, psi = nls_solve.solve(nls_t.initial_wave(x, "modulation", 1e-3, 1.0), half, dt, 20.0, n_frames=41)
    assert psi[0].abs().max() < 1.01
    assert psi.abs().max() > 2.5
    N, H = nls_eq.norm(psi, dx), nls_eq.hamiltonian(psi, k, dx)
    assert torch.allclose(N, N[0].expand_as(N), rtol=1e-10)
    assert (H - H[0]).abs().max().item() < 1e-3 * N[0].item()


# --- ripple tank --------------------------------------------------------------------------------------------------


def test_ripple_tank_cfl_enforced():
    assert rt_eq.time_step(1.0, 1.0, 0.7) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        rt_eq.time_step(1.0, 1.0, 0.75)


def test_ripple_tank_nodal_lines_at_half_odd_path_difference():
    lam = 16.0
    X, Y = rt_t.grid(CPU)
    gamma = rt_t.damping(CPU)
    blob, s1, s2 = rt_t.sources(X, Y, 3 * lam)
    dt = rt_eq.time_step(rt_t.H, rt_t.C, 0.5)
    _, _, intensity = rt_solve.solve(blob, gamma, rt_t.stencil(CPU), lam, rt_t.C, rt_t.H, dt, 300.0, n_frames=2)
    D = rt_eq.path_difference(X, Y, s1, s2)
    far = (gamma == 0) & (Y > rt_t.SOURCE_Y + 3 * lam)
    I_node, I_anti = rt_solve.nodal_contrast(intensity[-1], D, lam, far)
    assert I_node < 0.2 * I_anti
    # a cut along a row far from the sources: intensity minima sit near |r1 - r2| = (m + 1/2) λ
    row = int(rt_t.SOURCE_Y + 6 * lam)
    I_row, D_row = intensity[-1][:, row], D[:, row]
    a = torch.arange(rt_t.BORDER + 1, rt_t.N - rt_t.BORDER - 1)
    minima = a[(I_row[a] < I_row[a - 1]) & (I_row[a] < I_row[a + 1])]
    z = D_row[minima] / lam - 0.5
    assert len(minima) >= 4
    assert (z - torch.round(z)).abs().max() < 0.15
