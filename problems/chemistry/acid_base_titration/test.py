"""Physics checks, written first (TDD): what any correct acid–base titration code must satisfy.

Run with `pytest problems/chemistry/acid_base_titration/test.py`. Each test is a physical law, not an implementation
detail: charge balance holds, speciation fractions sum to one, pH = pKa at half-equivalence, and Henderson–Hasselbalch.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _titration(V, c_a=0.1, v0=25.0, c_b=0.1, pka_weak=4.76):
    _, pkas = tensors.acids(pka_weak)
    log_beta = tensors.log_beta(tensors.pka_table(pkas, CPU))
    C_A, Na = tensors.totals(V, c_a, v0, c_b)
    return solve.solve_ph(log_beta, C_A, Na, tensors.KW), log_beta


def test_strong_acid_equivalence_is_neutral_and_start_ph():
    V = torch.tensor([0.0, 25.0], dtype=torch.float64)
    pH, _ = _titration(V)
    assert abs(pH[0, 1].item() - 7.0) < 0.01  # HCl at equivalence: [H+] = [OH-] = √Kw
    assert abs(pH[0, 0].item() - 1.0) < 0.01  # 0.1 M strong acid
    # weak acid at the start: pH ≈ ½(pKa - log c) = 2.88
    assert abs(pH[1, 0].item() - 0.5 * (4.76 + 1.0)) < 0.02


def test_weak_acid_half_equivalence_equals_pka():
    for pka in (3.5, 4.76, 6.0):
        pH, _ = _titration(torch.tensor([12.5], dtype=torch.float64), pka_weak=pka)
        assert abs(pH[1, 0].item() - pka) < 0.01  # [HA] = [A-]  ⇒  pH = pKa


def test_charge_balance_residual_and_equivalence_points():
    V = tensors.volumes(80.0, 1601, CPU)
    pH, log_beta = _titration(V)
    C_A, Na = tensors.totals(V, 0.1, 25.0, 0.1)
    assert equation.charge_balance(pH, log_beta, C_A, Na, tensors.KW).abs().max() < 1e-12  # electroneutrality
    assert (pH.diff(dim=-1) > 0).all()  # adding base always raises the pH
    found = solve.equivalence_points(solve.slope(pH, V), V)
    assert [round(v) for v in found[0]] == [25]
    assert [round(v) for v in found[1]] == [25]
    assert [round(v) for v in found[2]] == [25, 50]  # H3PO4: the third jump is lost in the excess of base


def test_alpha_fractions_sum_to_one_and_cross_at_pka():
    _, pkas = tensors.acids(4.76)
    log_beta = tensors.log_beta(tensors.pka_table(pkas, CPU))
    pH = torch.linspace(0, 14, 141, dtype=torch.float64).expand(3, -1)
    alpha = equation.fractions(pH, log_beta)  # (3, 141, 4)
    assert torch.allclose(alpha.sum(-1), torch.ones(3, 141, dtype=torch.float64), atol=1e-12)  # mass balance
    assert (alpha[:2, :, 2:] == 0).all()  # monoprotic acids never lose a second proton
    # H3PO4: neighbouring species are equal at pH = pKa
    at_pka = equation.fractions(torch.tensor([[2.15, 7.20, 12.35]], dtype=torch.float64), log_beta[2:3])[0]
    for j in range(3):
        assert torch.isclose(at_pka[j, j], at_pka[j, j + 1], rtol=1e-9)


def test_henderson_hasselbalch_in_buffer_region():
    V = torch.linspace(5.0, 20.0, 31, dtype=torch.float64)
    pH, _ = _titration(V)
    hh = equation.henderson_hasselbalch(4.76, 0.1 * V, 0.1 * 25.0 - 0.1 * V)
    assert (pH[1] - hh).abs().max() < 0.02  # buffer limit: pH = pKa + log([A-]/[HA])
