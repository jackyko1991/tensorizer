"""Physics checks, written first (TDD): what any correct Lennard-Jones molecular-dynamics code must satisfy.

Run with `pytest problems/physchem/lj_liquid_rdf/test.py`. Each test is a physical law, not an implementation
detail: forces are −∇U and sum to zero, NVE energy is conserved, g(r) → 1, and a dilute gas obeys the virial equation.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _system(n_cells=4, rho_liquid=0.84, T_liquid=0.72):
    rho, T, T_start = tensors.state_points(rho_liquid, T_liquid, CPU)
    x, L = tensors.fcc_lattice(n_cells, rho)
    v = tensors.maxwell_velocities(x.shape, T_start, CPU)
    return x, v, L, rho, T, T_start


def test_forces_are_minus_gradient_and_sum_to_zero():
    x, v, L, *_ = _system()
    x = (x + 0.05 * torch.randn(x.shape, generator=torch.Generator().manual_seed(0))).double()
    L = L.double()
    f, U, _, _ = solve.evaluate(x, L)
    assert f.sum(1).abs().max() < 1e-9  # Newton's third law: total force is zero
    h = 1e-6
    for s, i, a in ((0, 3, 0), (1, 17, 2), (2, 100, 1)):  # central differences of U
        xp, xm = x.clone(), x.clone()
        xp[s, i, a] += h
        xm[s, i, a] -= h
        dU = (solve.evaluate(xp, L)[1][s] - solve.evaluate(xm, L)[1][s]) / (2 * h)
        assert abs(f[s, i, a].item() + dU.item()) < 1e-5 * max(1.0, abs(f[s, i, a].item()))  # conservative: F = −∇U


def test_nve_energy_conservation():
    x, v, L, rho, T, T_start = _system()
    energies = []

    def record(step, x, v, U, r2, f_over_r):
        energies.append((U + 0.5 * (v * v).sum((1, 2))) / x.shape[1])

    solve.verlet(x, v, L, 0.005, 400, record=record)  # no thermostat
    E = torch.stack(energies)  # (K, S)
    assert (E[-1] - E[0]).abs().max() < 2e-3  # isolated system: total energy is constant
    assert (E.max(0).values - E.min(0).values).max() < 5e-3


def test_rdf_tends_to_one_and_liquid_peak_near_minimum():
    x, v, L, rho, T, T_start = _system()
    x, v = solve.equilibrate(x, v, L, T, T_start, 0.005, 450)
    r_max = 0.5 * L.min().item()
    res = solve.production(x, v, L, rho, 0.005, 300, r_max, n_bins=60)
    r, g = res["r"], res["g"]
    peak = solve.first_peak(r, g)
    assert abs(peak[1].item() - 2 ** (1 / 6)) < 0.12  # liquid first shell near the LJ minimum
    assert g[1, 0:15].max() < 0.05  # atoms never overlap (r < 0.75 σ)
    far = r > 0.8 * r_max
    assert abs(g[1, far].mean().item() - 1) < 0.1  # g → 1 at large r in the liquid (no long-range order)
    assert abs(g[0, far].mean().item() - 1) < 0.15  # and in the gas


def test_dilute_gas_pressure_near_ideal_with_second_virial():
    rho, T = torch.tensor([0.02]), torch.tensor([2.0])
    x, L = tensors.fcc_lattice(4, rho)
    v = tensors.maxwell_velocities(x.shape, T, CPU)
    x, v = solve.verlet(x, v, L, 0.005, 300, T_target=T)
    P = []

    def record(step, x, v, U, r2, f_over_r):
        T_kin = equation.kinetic_temperature(v)
        P.append(equation.virial_pressure(r2, f_over_r, T_kin, rho, L) / (rho * T_kin))

    solve.verlet(x, v, L, 0.005, 300, record=record)
    Z = torch.stack(P).mean().item()
    B2 = equation.second_virial(2.0)
    assert abs(B2 - (-0.6276 * 2 * math.pi / 3)) < 0.03  # tabulated B₂*(T* = 2) = -0.6276 in units of 2πσ³/3
    assert abs(Z - 1) < 0.1  # ideal-gas limit: PV = NkT
    assert abs(Z - (1 + B2 * 0.02)) < 0.05  # virial equation of state: Z = 1 + B₂ ρ
