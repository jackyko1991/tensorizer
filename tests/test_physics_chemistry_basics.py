"""Checks for the beginner chemistry problems: equation balancing, titration curves, limiting reagent."""

import math

import torch

from problems.chemistry.acid_base_titration import equation as ab_eq
from problems.chemistry.acid_base_titration import solve as ab_solve
from problems.chemistry.acid_base_titration import tensors as ab_t
from problems.chemistry.equation_balancing import equation as eb_eq
from problems.chemistry.equation_balancing import solve as eb_solve
from problems.chemistry.equation_balancing import tensors as eb_t
from problems.chemistry.limiting_reagent_yield import equation as lr_eq
from problems.chemistry.limiting_reagent_yield import solve as lr_solve
from problems.chemistry.limiting_reagent_yield import tensors as lr_t

CPU = torch.device("cpu")


# ---------- equation balancing ----------


def _balance_all():
    names = list(eb_t.REACTIONS)
    parts = [eb_t.composition(*eb_t.REACTIONS[n], CPU) for n in names]
    M, n_species = eb_t.batch([eb_eq.signed_composition(C, s) for C, s, _, _ in parts])
    sv, V = eb_solve.null_vectors(M)
    nu, ok = eb_solve.to_integers(V[..., -1])
    return names, M, n_species, sv, V, nu, ok


def test_formula_parser():
    assert eb_t.parse_formula("H2O") == {"H": 2, "O": 1}
    assert eb_t.parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert eb_t.parse_formula("Ca3(PO4)2") == {"Ca": 3, "P": 2, "O": 8}
    assert eb_t.parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}
    assert eb_t.parse_formula("MnO4^-") == {"Mn": 1, "O": 4, "q": -1}
    assert eb_t.parse_formula("Fe^3+") == {"Fe": 1, "q": 3}
    assert eb_t.parse_formula("SO4^2-") == {"S": 1, "O": 4, "q": -2}


def test_known_balanced_coefficients():
    names, _, n_species, _, _, nu, ok = _balance_all()
    expected = {
        "methane": [1, 2, 1, 2],  # CH4 + 2 O2 → CO2 + 2 H2O
        "propane": [1, 5, 3, 4],  # C3H8 + 5 O2 → 3 CO2 + 4 H2O
        "glucose": [1, 6, 6, 6],  # C6H12O6 + 6 O2 → 6 CO2 + 6 H2O
        "photosynthesis": [6, 6, 1, 6],  # 6 CO2 + 6 H2O → C6H12O6 + 6 O2
        "permanganate": [1, 5, 8, 1, 5, 4],  # MnO4- + 5 Fe2+ + 8 H+ → Mn2+ + 5 Fe3+ + 4 H2O
        "phosphate": [3, 2, 1, 6],  # 3 Ca(OH)2 + 2 H3PO4 → Ca3(PO4)2 + 6 H2O
    }
    for name, coeffs in expected.items():
        r = names.index(name)
        assert ok[r]
        assert nu[r, : n_species[r]].long().tolist() == coeffs, name
        assert (nu[r, n_species[r] :] == 0).all()  # padding stays zero


def test_element_and_charge_conservation():
    names, M, _, _, _, nu, _ = _balance_all()
    residual = eb_eq.imbalance(M, nu)
    unique = [n != "peroxide" for n in names]
    assert (residual[unique] == 0).all()
    # the permanganate matrix has a charge row, and the charges really balance: -1 + 5·2 + 8·1 = 2 + 5·3 = 17
    C, sigma, elements, species = eb_t.composition(*eb_t.REACTIONS["permanganate"], CPU)
    assert elements[-1] == "q"
    charge = C[-1] * torch.tensor([1, 5, 8, 1, 5, 4], dtype=torch.float64)
    assert charge[:3].sum() == charge[3:].sum() == 17


def test_two_dimensional_null_space_detected():
    names, _, n_species, sv, V, _, _ = _balance_all()
    dims = eb_eq.null_dimension(sv)
    assert dims[names.index("peroxide")] == 2
    assert all(dims[i] == 1 for i, n in enumerate(names) if n != "peroxide")
    # the two basis reactions with one coefficient set to zero: 2 H2 + O2 → 2 H2O  and  H2 + O2 → H2O2
    r = names.index("peroxide")
    cands = eb_solve.reactions_in_plane(V[r, : n_species[r], -2:])
    physical = {tuple(c.long().tolist()) for c in cands if eb_solve.physical(c)}
    assert {(2, 1, 2, 0), (1, 1, 0, 1)} <= physical
    # without the charge row, the permanganate equation would not be unique either
    C, sigma, _, _ = eb_t.composition(*eb_t.REACTIONS["permanganate"], CPU)
    no_charge = eb_eq.signed_composition(C[:-1], sigma)
    sv_nc = torch.linalg.svdvals(torch.cat([no_charge, torch.zeros(2, 6, dtype=torch.float64)]))
    assert eb_eq.null_dimension(sv_nc[None]) == 2


# ---------- acid–base titration ----------


def _titration(V, c_a=0.1, v0=25.0, c_b=0.1, pka_weak=4.76):
    _, pkas = ab_t.acids(pka_weak)
    log_beta = ab_t.log_beta(ab_t.pka_table(pkas, CPU))
    C_A, Na = ab_t.totals(V, c_a, v0, c_b)
    return ab_solve.solve_ph(log_beta, C_A, Na, ab_t.KW), log_beta


def test_strong_acid_equivalence_is_neutral_and_start_ph():
    V = torch.tensor([0.0, 25.0], dtype=torch.float64)
    pH, _ = _titration(V)
    assert abs(pH[0, 1].item() - 7.0) < 0.01  # HCl at equivalence
    assert abs(pH[0, 0].item() - 1.0) < 0.01  # 0.1 M strong acid
    # weak acid at the start: pH ≈ ½(pKa - log c) = 2.88
    assert abs(pH[1, 0].item() - 0.5 * (4.76 + 1.0)) < 0.02


def test_weak_acid_half_equivalence_equals_pka():
    for pka in (3.5, 4.76, 6.0):
        pH, _ = _titration(torch.tensor([12.5], dtype=torch.float64), pka_weak=pka)
        assert abs(pH[1, 0].item() - pka) < 0.01


def test_charge_balance_residual_and_equivalence_points():
    V = ab_t.volumes(80.0, 1601, CPU)
    pH, log_beta = _titration(V)
    C_A, Na = ab_t.totals(V, 0.1, 25.0, 0.1)
    assert ab_eq.charge_balance(pH, log_beta, C_A, Na, ab_t.KW).abs().max() < 1e-12
    assert (pH.diff(dim=-1) > 0).all()  # adding base always raises the pH
    found = ab_solve.equivalence_points(ab_solve.slope(pH, V), V)
    assert [round(v) for v in found[0]] == [25]
    assert [round(v) for v in found[1]] == [25]
    assert [round(v) for v in found[2]] == [25, 50]  # H3PO4: the third jump is lost in the excess of base


def test_alpha_fractions_sum_to_one_and_cross_at_pka():
    _, pkas = ab_t.acids(4.76)
    log_beta = ab_t.log_beta(ab_t.pka_table(pkas, CPU))
    pH = torch.linspace(0, 14, 141, dtype=torch.float64).expand(3, -1)
    alpha = ab_eq.fractions(pH, log_beta)  # (3, 141, 4)
    assert torch.allclose(alpha.sum(-1), torch.ones(3, 141, dtype=torch.float64), atol=1e-12)
    assert (alpha[:2, :, 2:] == 0).all()  # monoprotic acids never lose a second proton
    # H3PO4: neighbouring species are equal at pH = pKa
    at_pka = ab_eq.fractions(torch.tensor([[2.15, 7.20, 12.35]], dtype=torch.float64), log_beta[2:3])[0]
    for j in range(3):
        assert torch.isclose(at_pka[j, j], at_pka[j, j + 1], rtol=1e-9)


def test_henderson_hasselbalch_in_buffer_region():
    V = torch.linspace(5.0, 20.0, 31, dtype=torch.float64)
    pH, _ = _titration(V)
    hh = ab_eq.henderson_hasselbalch(4.76, 0.1 * V, 0.1 * 25.0 - 0.1 * V)
    assert (pH[1] - hh).abs().max() < 0.02


# ---------- limiting reagent ----------


def test_molar_masses():
    _, molar, _ = lr_t.stoichiometry("aluminium_iodine", CPU)
    assert torch.allclose(molar, torch.tensor([26.982, 253.808, 407.694], dtype=torch.float64))
    _, molar, _ = lr_t.stoichiometry("haber", CPU)
    assert torch.allclose(molar, torch.tensor([28.014, 2.016, 17.031], dtype=torch.float64))


def test_limiting_reagent_against_hand_calculation():
    nu, molar, _ = lr_t.stoichiometry("aluminium_iodine", CPU)
    m0, weighed = lr_t.experiments("aluminium_iodine", 1.20, 2.40, 2.00, CPU)
    res = lr_solve.analyse(m0, weighed, nu, molar)
    # by hand: Al 1.20/26.982 = 44.47 mmol → /2 = 22.24;  I2 2.40/253.808 = 9.456 mmol → /3 = 3.152  ⇒ I2 limiting
    n_al, n_i2 = 1.20 / 26.982, 2.40 / 253.808
    xi = n_i2 / 3
    assert res["limiting"][-1].item() == 1
    assert math.isclose(res["xi_max"][-1].item(), xi, rel_tol=1e-12)
    assert math.isclose(res["theory"][-1].item(), 2 * xi * 407.694, rel_tol=1e-12)  # 2.570 g AlI3
    assert abs(res["theory"][-1].item() - 2.570) < 1e-3
    assert math.isclose(res["leftover_mass"][-1, 0].item(), (n_al - 2 * xi) * 26.982, rel_tol=1e-12)  # 1.030 g Al
    assert abs(res["leftover_mass"][-1, 1].item()) < 1e-15  # no iodine left
    assert math.isclose(res["percent"][-1].item(), 100 * 2.00 / (2 * xi * 407.694), rel_tol=1e-12)  # 77.8 %


def test_haber_limiting_and_mass_conservation():
    nu, molar, _ = lr_t.stoichiometry("haber", CPU)
    m0 = torch.tensor([[28.0, 10.0, 0.0], [28.0, 3.0, 0.0]], dtype=torch.float64)
    res = lr_solve.analyse(m0, torch.tensor([5.0, 5.0], dtype=torch.float64), nu, molar)
    # 1 mol N2 with 4.96 mol H2: N2 limiting, 2 mol NH3;  with 1.49 mol H2: H2 limiting, 0.992 mol NH3
    assert res["limiting"].tolist() == [0, 1]
    assert math.isclose(res["theory"][0].item(), 2 * 28.0 / 28.014 * 17.031, rel_tol=1e-12)
    assert math.isclose(res["theory"][1].item(), 2 * (3.0 / 2.016 / 3) * 17.031, rel_tol=1e-12)
    # total mass is conserved at every extent (atomic masses add up exactly)
    xi, n = lr_solve.extent_sweep(res["n0"], nu, res["xi_max"])
    mass = torch.einsum("bks,s->bk", n, molar)
    assert torch.allclose(mass, m0.sum(-1, keepdim=True).expand_as(mass), rtol=1e-12)
    assert (n >= -1e-12).all()


def test_yield_map_matches_batch_and_stoichiometric_line():
    nu, molar, _ = lr_t.stoichiometry("aluminium_iodine", CPU)
    g1 = torch.linspace(0.0, 2.0, 21, dtype=torch.float64)
    g2 = torch.linspace(0.0, 28.0, 15, dtype=torch.float64)
    pm = lr_solve.yield_map(g1, g2, nu, molar)
    assert pm.shape == (15, 21)
    xi, _ = lr_eq.max_extent(torch.stack([g1 / molar[0], torch.full_like(g1, 14.0) / molar[1], 0 * g1], -1), nu)
    assert torch.allclose(pm[7], lr_eq.theoretical_mass(xi, nu, molar))
    # on the stoichiometric line both reactants run out together: product mass = m_Al + m_I2
    ratio = (3 * molar[1]) / (2 * molar[0])
    m_al = torch.tensor([0.5, 1.0], dtype=torch.float64)
    n0 = torch.stack([m_al / molar[0], ratio * m_al / molar[1], torch.zeros(2, dtype=torch.float64)], -1)
    xi, _ = lr_eq.max_extent(n0, nu)
    assert torch.allclose(lr_eq.theoretical_mass(xi, nu, molar), m_al * (1 + ratio), rtol=1e-10)
