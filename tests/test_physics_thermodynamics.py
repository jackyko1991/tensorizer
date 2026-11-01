import math

import pytest
import torch

from problems.thermodynamics.blackbody_radiation import equation as bb_eq
from problems.thermodynamics.blackbody_radiation import solve as bb_solve
from problems.thermodynamics.blackbody_radiation import tensors as bb_t
from problems.thermodynamics.carnot_cycle import solve as carnot_solve
from problems.thermodynamics.carnot_cycle import tensors as carnot_t
from problems.thermodynamics.heat_diffusion import solve as heat_solve
from problems.thermodynamics.heat_diffusion import tensors as heat_t
from problems.thermodynamics.ideal_gas_processes import equation as ig_eq
from problems.thermodynamics.ideal_gas_processes import solve as ig_solve
from problems.thermodynamics.ideal_gas_processes import tensors as ig_t
from problems.thermodynamics.ising_model import equation as ising_eq
from problems.thermodynamics.ising_model import solve as ising_solve
from problems.thermodynamics.ising_model import tensors as ising_t
from problems.thermodynamics.kinetic_gas import equation as gas_eq
from problems.thermodynamics.kinetic_gas import solve as gas_solve
from problems.thermodynamics.kinetic_gas import tensors as gas_t
from problems.thermodynamics.newton_cooling import equation as cool_eq
from problems.thermodynamics.newton_cooling import solve as cool_solve
from problems.thermodynamics.newton_cooling import tensors as cool_t
from problems.thermodynamics.random_walk_diffusion import solve as walk_solve
from problems.thermodynamics.random_walk_diffusion import tensors as walk_t
from problems.thermodynamics.van_der_waals import equation as vdw_eq
from problems.thermodynamics.van_der_waals import solve as vdw_solve
from problems.thermodynamics.van_der_waals import tensors as vdw_t

CPU = torch.device("cpu")


def test_heat_sine_mode_decays_at_analytic_rate():
    X, Y, dx = heat_t.grid(48, CPU)
    T0 = heat_t.sine_mode(X, Y).double()
    alpha, t_max = 2e-5, 20.0
    times, frames = heat_solve.solve(T0, heat_t.K, heat_t.heat_capacity(alpha), dx, t_max, boundary="dirichlet")
    rate = alpha * 2 * (math.pi / heat_t.PLATE) ** 2  # α k² with k² = kx² + ky²
    amplitude = frames[-1].max() / frames[0].max()
    assert math.isclose(amplitude.item(), math.exp(-rate * t_max), rel_tol=1e-2)
    # the shape is an eigenmode: it only shrinks, it does not deform
    assert torch.allclose(frames[-1] / amplitude, frames[0], atol=1e-6)


def test_heat_is_conserved_with_insulated_edges():
    X, Y, dx = heat_t.grid(32, CPU)
    T0 = heat_t.hot_spots(X, Y, 3).double()
    rho_c = heat_t.heat_capacity(5e-5)
    _, frames = heat_solve.solve(T0, heat_t.K, rho_c, dx, 10.0, boundary="neumann", T_edge=heat_t.T_EDGE)
    heat = heat_solve.heat_content(frames, rho_c, dx, heat_t.T_EDGE)
    assert torch.allclose(heat, heat[0].expand_as(heat), rtol=1e-10)
    assert frames[-1].max() < frames[0].max()  # yet the hot spots spread out


def test_heat_rejects_unstable_time_step():
    with pytest.raises(ValueError):
        heat_solve.stable_dt(1e-4, 1e-3, 0.3)


def run_gas(n=200, radius=0.005, t_max=2.0):
    x0 = gas_t.initial_positions(n, radius, CPU)
    v0 = gas_t.initial_velocities(n, 1.0, CPU)
    return gas_solve.solve(x0, v0, gas_t.BOX, radius, t_max), v0


def test_gas_conserves_kinetic_energy_and_follows_ideal_gas_law():
    (times, xs, vs, pressure), v0 = run_gas()
    energy = gas_eq.kinetic_energy(vs)
    assert torch.allclose(energy, energy[0].expand_as(energy), rtol=1e-4)

    kT = gas_eq.temperature(v0).item()
    area = (gas_t.BOX - 2 * 0.005) ** 2
    assert math.isclose(pressure, 200 * kT / area, rel_tol=0.15)


def test_gas_speeds_relax_away_from_a_single_speed():
    (_, _, vs, _), v0 = run_gas()
    speed = vs[-1].norm(dim=1)
    kT = gas_eq.temperature(v0).item()
    assert speed.std() > 0.3  # started at exactly 0
    assert math.isclose(speed.mean().item(), math.sqrt(math.pi * kT / 2), rel_tol=0.1)  # 2D Maxwell–Boltzmann mean


def test_carnot_efficiency_and_first_law():
    for gas in ["monatomic", "diatomic"]:
        gamma = carnot_t.GAMMA[gas]
        legs = carnot_t.cycle(n=1.0, T_h=600.0, T_c=300.0, ratio=2.5, gamma=gamma)
        r = carnot_solve.solve(legs, 1.0, 600.0, 300.0, gamma)
        assert math.isclose(r["efficiency"], 1 - 300.0 / 600.0, rel_tol=1e-4)
        assert math.isclose(r["W"], r["Q_h"] - r["Q_c"], rel_tol=1e-4)
        # the two adiabats cancel: equal and opposite work
        assert math.isclose(r["W_legs"][1].item(), -r["W_legs"][3].item(), rel_tol=1e-6)


def test_newton_cooling_matches_exponential_and_half_life():
    k = cool_t.cooling_constants(4, 0.02, 0.2, CPU).double()
    T0 = cool_t.start_temperatures(4, CPU).double()
    times, temps = cool_solve.solve(T0, k, 20.0, 60.0)
    exact = cool_eq.exact_temperature(times, T0, k, 20.0)  # (T, B)
    assert torch.allclose(temps, exact, atol=1e-6)

    # after one half-life the gap to the room has halved, whatever the cup
    t_half = cool_eq.half_life(k)  # (B,)
    gap = cool_eq.exact_temperature(t_half, T0, k, 20.0).diagonal() - 20.0  # cup b at its own t½
    assert torch.allclose(gap, (T0 - 20.0) / 2, rtol=1e-10)
    assert torch.allclose(t_half, math.log(2) / k)


def test_ideal_gas_process_work_and_first_law():
    n, T0, ratio, gamma = 1.0, 300.0, 3.0, 5 / 3
    paths = ig_t.processes(n, T0, ratio, gamma, points=2000)
    r = ig_solve.solve(paths, n, gamma)
    W_iso, W_bar, W_cho, W_adi = r["W"].tolist()

    assert math.isclose(W_iso, n * ig_eq.R * T0 * math.log(ratio), rel_tol=1e-5)
    assert W_cho == 0.0  # dV = 0 on the isochore
    p1 = n * ig_eq.R * T0 / ig_t.V1
    assert math.isclose(W_bar, p1 * (ratio - 1) * ig_t.V1, rel_tol=1e-10)

    V, p = paths[3, :, 0], paths[3, :, 1]  # adiabat: p V^γ stays constant and Q = 0
    pVg = p * V**gamma
    assert torch.allclose(pVg, pVg[0].expand_as(pVg), rtol=1e-10)
    assert abs(r["Q"][3].item()) < 1e-4 * abs(W_adi)
    assert math.isclose(r["Q"][0].item(), W_iso, rel_tol=1e-8)  # isotherm: ΔU = 0, so Q = W


def test_random_walk_msd_grows_as_2dDt():
    D, t_max, n_steps = 0.7, 5.0, 100
    t = walk_t.times(t_max, n_steps, CPU)
    kicks = walk_t.random_kicks(n_steps, 5000, D, t_max / n_steps, CPU, seed=3)
    r = walk_solve.solve(t, kicks)
    assert math.isclose(r["D_fit"], D, rel_tol=0.05)  # MSD slope = 2dD with d = 2

    var = r["x"][-1].var(dim=0)  # each axis spreads to variance 2Dt
    assert torch.allclose(var, torch.full_like(var, 2 * D * t_max), rtol=0.08)
    assert r["x"][-1].mean(dim=0).abs().max() < 0.1 * math.sqrt(2 * D * t_max)  # no drift


def test_blackbody_wien_peak_and_stefan_boltzmann():
    T = bb_t.temperatures(3000.0, 10000.0, 8, CPU)
    lam = bb_t.wavelengths(CPU)
    r = bb_solve.solve(T, lam)
    assert r["u"].shape == (8, len(lam))

    wien = r["lam_max"] * T  # λ_max T = b for every temperature
    assert torch.allclose(wien, torch.full_like(wien, 2.898e-3), rtol=5e-3)

    U = r["U"]
    assert torch.allclose(U / U[0], (T / T[0]) ** 4, rtol=1e-4)  # ratio test: U ∝ T⁴
    assert torch.allclose(U, bb_eq.stefan_boltzmann(T), rtol=1e-4)  # and the prefactor 4σ/c


def run_ising(temps, L=16, n_sweeps=300, seed=0):
    T = torch.tensor(temps)
    s0 = ising_t.random_spins(len(T), L, CPU, seed=seed)
    return ising_solve.solve(s0, T, ising_t.checkerboard(L, CPU), n_sweeps)


def test_ising_orders_when_cold_and_disorders_when_hot():
    r = run_ising([0.5, 1.0, 10.0])
    m = r["m"][-50:].abs().mean(dim=0)  # (B,)
    # a cold quench can be trapped in two stripe domains, so ask that one of the cold lattices orders fully
    assert max(m[0].item(), m[1].item()) > 0.95
    assert m[2] < 0.15
    assert r["E"][-1, 2] > -0.5  # hot: hardly more aligned bonds than anti-aligned ones


def test_ising_energy_conv_matches_explicit_bond_sum():
    s = ising_t.random_spins(3, 12, CPU, seed=5)
    # every bond counted once: right and down neighbours, wrapping around with torch.roll
    right = torch.roll(s, shifts=-1, dims=-1)
    down = torch.roll(s, shifts=-1, dims=-2)
    explicit = -(s * right + s * down).sum(dim=(1, 2, 3)) / 144
    assert torch.allclose(ising_eq.energy_per_spin(s), explicit)

    ordered = torch.ones(1, 1, 8, 8)
    assert ising_eq.energy_per_spin(ordered).item() == -2.0
    assert torch.equal(ising_eq.neighbor_sum(ordered), 4 * ordered)


def test_van_der_waals_critical_point():
    a, b = vdw_t.GASES["CO2"]
    n = 2.0
    T_c, V_c, p_c = vdw_eq.critical_point(n, a, b)
    assert math.isclose(V_c, 3 * n * b) and math.isclose(p_c, a / (27 * b**2))
    assert math.isclose(T_c, 8 * a / (27 * vdw_eq.R * b))

    # the critical isotherm passes through (V_c, p_c) with zero slope and zero curvature
    h = 1e-3 * V_c
    V = torch.tensor([V_c - h, V_c, V_c + h], dtype=torch.float64)
    p = vdw_eq.pressure(V, torch.tensor([T_c], dtype=torch.float64), n, a, b)[0]
    assert math.isclose(p[1].item(), p_c, rel_tol=1e-10)
    assert abs((p[2] - p[0]).item() / (2 * h)) * V_c / p_c < 1e-5
    assert abs((p[2] - 2 * p[1] + p[0]).item() / h**2) * V_c**2 / p_c < 1e-3


def test_van_der_waals_maxwell_equal_areas():
    a, b = vdw_t.GASES["water"]
    n = 1.0
    T_c, V_c, p_c = vdw_eq.critical_point(n, a, b)
    T = vdw_t.temperatures(T_c, 1.1, 0.75, 8)
    V = vdw_t.volumes(V_c)
    r = vdw_solve.solve(T, V, T_c, V_c, n, a, b)
    below = T < T_c
    assert torch.isnan(r["p_star"][~below]).all()

    for t in torch.nonzero(below).flatten().tolist():
        p_star, V_l, V_g = r["p_star"][t].item(), r["V_l"][t].item(), r["V_g"][t].item()
        assert V_l < V_c < V_g and 0 < p_star < p_c
        # independent check on a fine grid: the signed area between curve and line vanishes
        V_fine = torch.linspace(V_l, V_g, 20001, dtype=torch.float64)
        gap = vdw_eq.pressure(V_fine, T[t : t + 1], n, a, b)[0] - p_star
        signed, total = torch.trapezoid(gap, V_fine), torch.trapezoid(gap.abs(), V_fine)
        assert abs(signed.item()) < 1e-4 * total.item()
        assert abs(gap[0].item()) < 1e-6 * p_c and abs(gap[-1].item()) < 1e-6 * p_c  # line ends on the curve

    # reduced vapour pressure of any van der Waals fluid at T = 0.8 T_c is about 0.383 p_c
    p_08 = vdw_solve.coexistence_pressure(torch.tensor([0.8 * T_c], dtype=torch.float64), V, V_c, n, a, b)
    assert math.isclose(p_08.item() / p_c, 0.3834, rel_tol=2e-3)
