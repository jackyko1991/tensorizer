"""Physics checks for the advanced physical-chemistry problems: Hartree–Fock H₂, Marcus theory, Lennard-Jones MD."""

import math

import torch

from problems.physchem.hartree_fock_h2 import equation as hf_eq
from problems.physchem.hartree_fock_h2 import solve as hf_solve
from problems.physchem.hartree_fock_h2 import tensors as hf_t
from problems.physchem.lj_liquid_rdf import equation as lj_eq
from problems.physchem.lj_liquid_rdf import solve as lj_solve
from problems.physchem.lj_liquid_rdf import tensors as lj_t
from problems.physchem.marcus_electron_transfer import equation as mc_eq
from problems.physchem.marcus_electron_transfer import solve as mc_solve
from problems.physchem.marcus_electron_transfer import tensors as mc_t

CPU = torch.device("cpu")


# ---------------------------------------------------------------- Hartree–Fock


def _hf(name="H2", R=(1.4,)):
    Z, zeta, n_atom = hf_t.molecule(name, CPU)
    alpha, coef = hf_t.sto3g(zeta)
    nuclei = hf_t.geometry(torch.tensor(R, dtype=torch.float64))
    ints = hf_solve.integrals(alpha, coef, nuclei, Z)
    res = hf_solve.scf(ints, hf_t.N_ELECTRONS)
    E = res["E_el"] + hf_eq.nuclear_repulsion(nuclei, Z)
    return ints, res, E, (alpha, coef, Z, n_atom)


def test_hf_overlap_normalised_and_positive_definite():
    ints, *_ = _hf(R=(0.7, 1.4, 3.0))
    S = ints["S"]
    assert torch.allclose(S.diagonal(dim1=-2, dim2=-1), torch.ones(3, 2, dtype=S.dtype), atol=1e-5)
    assert torch.allclose(S, S.transpose(-1, -2))
    assert torch.linalg.eigvalsh(S).min() > 0
    assert abs(S[1, 0, 1].item() - 0.6593) < 1e-3  # Szabo & Ostlund S₁₂ at R = 1.4 bohr


def test_hf_h2_energy_matches_szabo_ostlund():
    ints, res, E, _ = _hf()
    assert abs(E[0].item() - (-1.1167)) < 1e-3
    assert abs(res["E_el"][0].item() - (-1.8310)) < 1e-3


def test_hf_fock_symmetric_converged_and_two_electrons():
    for name, R in (("H2", 1.4), ("HeH+", 1.4632)):
        ints, res, E, _ = _hf(name, (R, 2.5))
        F, P, S = res["F"], res["P"], ints["S"]
        assert res["converged"]
        assert torch.allclose(F, F.transpose(-1, -2), atol=1e-12)
        n = torch.einsum("bmn,bnm->b", P, S)
        assert torch.allclose(n, torch.full_like(n, 2.0), atol=1e-8)
        # self-consistency: P built from the eigenvectors of F(P) reproduces F
        assert torch.allclose(hf_eq.fock(ints["H"], P, ints["eri"]), F, atol=1e-10)
    _, _, E, _ = _hf("HeH+", (1.4632,))
    assert abs(E[0].item() - (-2.8607)) < 1e-3


def test_hf_dissociation_failure_and_full_ci_limit():
    ints, res, E, (alpha, coef, Z, n_atom) = _hf(R=(1.4, 10.0))
    E_nuc = hf_eq.nuclear_repulsion(hf_t.geometry(torch.tensor([1.4, 10.0], dtype=torch.float64)), Z)
    E_ci = hf_solve.full_ci(ints, res["C"]) + E_nuc
    limit = hf_solve.separated_atoms(alpha, coef, Z, n_atom)
    assert abs(limit - 2 * (-0.4666)) < 1e-3  # two STO-3G hydrogen atoms
    assert E[1] - limit > 0.2  # RHF stays far above H + H
    assert abs(E_ci[1].item() - limit) < 1e-3  # full CI dissociates correctly
    assert -0.03 < (E_ci[0] - E[0]).item() < -0.01  # correlation energy ≈ -0.02 hartree


# ---------------------------------------------------------------- Marcus


def test_marcus_rate_peaks_at_minus_lambda_and_inverted_region_falls():
    dG = mc_t.driving_forces(-2.0, 0.5, 2501, CPU).double()
    lams = torch.tensor([0.3, 0.6, 1.0], dtype=torch.float64)
    kT = mc_t.KB * torch.tensor([250.0, 300.0], dtype=torch.float64)
    k = mc_solve.rate_grid(dG, lams, kT, 0.01)  # (L, T, G)
    for i, lam in enumerate(lams.tolist()):
        for t in range(len(kT)):
            g = int(k[i, t].argmax())
            assert abs(dG[g].item() + lam) < 2e-3
            inverted = k[i, t, :g]  # more negative ΔG° than -λ
            assert (inverted[1:] - inverted[:-1] >= 0).all()  # rate falls as the driving force grows
            normal = k[i, t, g:]
            assert (normal[1:] - normal[:-1] <= 0).all()
    # symmetric about -λ, and the activationless rate is the prefactor
    lam, kT0 = 0.6, mc_t.KB * 300
    a = mc_eq.marcus_rate(torch.tensor(-lam - 0.2), torch.tensor(lam), torch.tensor(kT0), 0.01)
    b = mc_eq.marcus_rate(torch.tensor(-lam + 0.2), torch.tensor(lam), torch.tensor(kT0), 0.01)
    assert torch.allclose(a, b)
    top = mc_eq.marcus_rate(torch.tensor(-lam), torch.tensor(lam), torch.tensor(kT0), 0.01)
    assert math.isclose(top.item(), 2 * math.pi / mc_t.HBAR * 1e-4 / math.sqrt(4 * math.pi * lam * kT0), rel_tol=1e-5)


def test_marcus_crossing_point_on_both_parabolas():
    lam, dG = 0.7, torch.linspace(-1.5, 0.3, 7)
    x = mc_eq.crossing_point(lam, dG)
    assert torch.allclose(mc_eq.reactant_surface(x, lam), mc_eq.product_surface(x, lam, dG), atol=1e-6)
    assert torch.allclose(mc_eq.reactant_surface(x, lam), mc_eq.activation_energy(lam, dG), atol=1e-6)


def test_marcus_surface_hopping_follows_the_rate():
    lam, kT, H, omega = 0.5, mc_t.KB * 300, 0.005, 20.0
    dG = torch.tensor([-0.25, -0.5, -0.75])
    x, v = mc_t.thermal_ensemble(3, 1000, lam, kT, omega, CPU)
    res = mc_solve.hopping_dynamics(x, v, dG, lam, kT, H, omega, 10.0, 0.1 / omega, 1500, [1500])
    k_sim, _ = mc_solve.simulated_rates(res)
    k_th = mc_eq.marcus_rate(dG, torch.tensor(lam), torch.tensor(kT), H)
    assert ((k_sim / k_th).log10().abs() < 0.15).all()
    assert k_sim[1] > k_sim[0] and k_sim[1] > k_sim[2]  # fastest at -ΔG° = λ, slower in the inverted region


# ---------------------------------------------------------------- Lennard-Jones MD


def _lj_system(n_cells=4, rho_liquid=0.84, T_liquid=0.72):
    rho, T, T_start = lj_t.state_points(rho_liquid, T_liquid, CPU)
    x, L = lj_t.fcc_lattice(n_cells, rho)
    v = lj_t.maxwell_velocities(x.shape, T_start, CPU)
    return x, v, L, rho, T, T_start


def test_lj_forces_are_minus_gradient_and_sum_to_zero():
    x, v, L, *_ = _lj_system()
    x = (x + 0.05 * torch.randn(x.shape, generator=torch.Generator().manual_seed(0))).double()
    L = L.double()
    f, U, _, _ = lj_solve.evaluate(x, L)
    assert f.sum(1).abs().max() < 1e-9  # Newton's third law
    h = 1e-6
    for s, i, a in ((0, 3, 0), (1, 17, 2), (2, 100, 1)):  # central differences of U
        xp, xm = x.clone(), x.clone()
        xp[s, i, a] += h
        xm[s, i, a] -= h
        dU = (lj_solve.evaluate(xp, L)[1][s] - lj_solve.evaluate(xm, L)[1][s]) / (2 * h)
        assert abs(f[s, i, a].item() + dU.item()) < 1e-5 * max(1.0, abs(f[s, i, a].item()))


def test_lj_nve_energy_conservation():
    x, v, L, rho, T, T_start = _lj_system()
    energies = []

    def record(step, x, v, U, r2, f_over_r):
        energies.append((U + 0.5 * (v * v).sum((1, 2))) / x.shape[1])

    lj_solve.verlet(x, v, L, 0.005, 400, record=record)  # no thermostat
    E = torch.stack(energies)  # (K, S)
    assert (E[-1] - E[0]).abs().max() < 2e-3
    assert (E.max(0).values - E.min(0).values).max() < 5e-3


def test_lj_rdf_tends_to_one_and_liquid_peak_near_minimum():
    x, v, L, rho, T, T_start = _lj_system()
    x, v = lj_solve.equilibrate(x, v, L, T, T_start, 0.005, 450)
    r_max = 0.5 * L.min().item()
    res = lj_solve.production(x, v, L, rho, 0.005, 300, r_max, n_bins=60)
    r, g = res["r"], res["g"]
    peak = lj_solve.first_peak(r, g)
    assert abs(peak[1].item() - 2 ** (1 / 6)) < 0.12  # liquid first shell near the LJ minimum
    assert g[1, 0:15].max() < 0.05  # atoms never overlap (r < 0.75 σ)
    far = r > 0.8 * r_max
    assert abs(g[1, far].mean().item() - 1) < 0.1  # g → 1 at large r in the liquid
    assert abs(g[0, far].mean().item() - 1) < 0.15  # and in the gas


def test_lj_dilute_gas_pressure_near_ideal_with_second_virial():
    rho, T = torch.tensor([0.02]), torch.tensor([2.0])
    x, L = lj_t.fcc_lattice(4, rho)
    v = lj_t.maxwell_velocities(x.shape, T, CPU)
    x, v = lj_solve.verlet(x, v, L, 0.005, 300, T_target=T)
    P = []

    def record(step, x, v, U, r2, f_over_r):
        P.append(
            lj_eq.virial_pressure(r2, f_over_r, lj_eq.kinetic_temperature(v), rho, L)
            / (rho * lj_eq.kinetic_temperature(v))
        )

    lj_solve.verlet(x, v, L, 0.005, 300, record=record)
    Z = torch.stack(P).mean().item()
    B2 = lj_eq.second_virial(2.0)
    assert abs(B2 - (-0.6276 * 2 * math.pi / 3)) < 0.03  # tabulated B₂*(T* = 2) = -0.6276 in units of 2πσ³/3
    assert abs(Z - 1) < 0.1
    assert abs(Z - (1 + B2 * 0.02)) < 0.05
