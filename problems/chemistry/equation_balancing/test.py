"""Physics checks, written first (TDD): what any correct equation-balancing code must satisfy.

Run with `pytest problems/chemistry/equation_balancing/test.py`. Each test is a physical law, not an implementation
detail: balanced equations conserve every element and the total charge, and a 2-D null space means no unique answer.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _balance_all():
    names = list(tensors.REACTIONS)
    parts = [tensors.composition(*tensors.REACTIONS[n], CPU) for n in names]
    M, n_species = tensors.batch([equation.signed_composition(C, s) for C, s, _, _ in parts])
    sv, V = solve.null_vectors(M)
    nu, ok = solve.to_integers(V[..., -1])
    return names, M, n_species, sv, V, nu, ok


def test_formula_parser():
    assert tensors.parse_formula("H2O") == {"H": 2, "O": 1}
    assert tensors.parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert tensors.parse_formula("Ca3(PO4)2") == {"Ca": 3, "P": 2, "O": 8}
    assert tensors.parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}
    assert tensors.parse_formula("MnO4^-") == {"Mn": 1, "O": 4, "q": -1}
    assert tensors.parse_formula("Fe^3+") == {"Fe": 1, "q": 3}
    assert tensors.parse_formula("SO4^2-") == {"S": 1, "O": 4, "q": -2}


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
    residual = equation.imbalance(M, nu)
    unique = [n != "peroxide" for n in names]
    assert (residual[unique] == 0).all()  # conservation of mass: every element balances exactly
    # the permanganate matrix has a charge row, and the charges really balance: -1 + 5·2 + 8·1 = 2 + 5·3 = 17
    C, sigma, elements, species = tensors.composition(*tensors.REACTIONS["permanganate"], CPU)
    assert elements[-1] == "q"
    charge = C[-1] * torch.tensor([1, 5, 8, 1, 5, 4], dtype=torch.float64)
    assert charge[:3].sum() == charge[3:].sum() == 17  # conservation of charge


def test_two_dimensional_null_space_detected():
    names, _, n_species, sv, V, _, _ = _balance_all()
    dims = equation.null_dimension(sv)
    assert dims[names.index("peroxide")] == 2
    assert all(dims[i] == 1 for i, n in enumerate(names) if n != "peroxide")
    # the two basis reactions with one coefficient set to zero: 2 H2 + O2 → 2 H2O  and  H2 + O2 → H2O2
    r = names.index("peroxide")
    cands = solve.reactions_in_plane(V[r, : n_species[r], -2:])
    physical = {tuple(c.long().tolist()) for c in cands if solve.physical(c)}
    assert {(2, 1, 2, 0), (1, 1, 0, 1)} <= physical
    # without the charge row, the permanganate equation would not be unique either
    C, sigma, _, _ = tensors.composition(*tensors.REACTIONS["permanganate"], CPU)
    no_charge = equation.signed_composition(C[:-1], sigma)
    sv_nc = torch.linalg.svdvals(torch.cat([no_charge, torch.zeros(2, 6, dtype=torch.float64)]))
    assert equation.null_dimension(sv_nc[None]) == 2
