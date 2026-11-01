"""Physics checks, written first (TDD): what any correct Gibbs-energy minimisation code must satisfy.

Run with `pytest problems/chemistry/gibbs_minimisation/test.py`. Each test is a physical law, not an implementation
detail: elements are conserved, G is a constrained minimum, hot products dissociate, and enthalpy balances the flame.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def _sweep(temps, phi=1.0, P=1.0):
    A, th = tensors.element_matrix(CPU), tensors.thermo(CPU)
    T = torch.tensor(temps, dtype=F64)
    b = tensors.feed(torch.tensor([phi], dtype=F64))[0]
    out = solve.temperature_sweep(T, A, b, P, th, n_iter=80)
    return A, th, T, b, out


def test_element_balance_and_lagrange_conditions_hold_at_the_optimum():
    A, th, T, b, out = _sweep([1200.0, 1800.0, 2400.0, 3000.0, 3500.0])
    g = equation.standard_potential(T, th)
    balance, kkt = solve.residuals(A, b.expand(len(T), -1), out["n"], out["pi"], g, 1.0)
    assert balance < 1e-9  # element conservation: A n = b
    assert kkt < 1e-8  # μ_j / RT = π_i A_ij
    assert (out["n"] > 0).all()


def test_gibbs_energy_is_lower_than_at_nearby_feasible_points():
    A, th, T, b, out = _sweep([2500.0])
    n = out["n"]  # (1, S), every species well above zero at 2500 K
    g = equation.standard_potential(T, th)
    G0 = equation.gibbs(n, g, 1.0).item()
    null = torch.linalg.svd(A).Vh[A.shape[0] :]  # (S - E, S): directions that keep A n = b
    gen = torch.Generator().manual_seed(0)
    for _ in range(25):
        d = torch.randn(null.shape[0], generator=gen, dtype=F64) @ null  # (S,)
        for frac in (0.3, 0.03):
            step = frac * (n[0] / d.abs()).min()  # keeps every n_j > 0
            n_new = n + step * d
            assert torch.allclose(equation.element_totals(A, n_new), b[None], atol=1e-10)  # still feasible
            assert equation.gibbs(n_new, g, 1.0).item() > G0  # second law: equilibrium minimises G at fixed T, P


def test_autograd_gradient_equals_element_potentials():
    A, th, T, b, out = _sweep([2200.0])
    g = equation.standard_potential(T, th)
    n = out["n"].clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(equation.gibbs(n, g, 1.0).sum(), n)
    assert torch.allclose(grad, out["pi"] @ A, atol=1e-8)


def test_products_dissociate_at_high_temperature():
    temps = [1200.0, 1600.0, 2000.0, 2400.0, 2800.0, 3200.0]
    names = tensors.SPECIES
    *_, out = _sweep(temps)
    x = equation.mole_fractions(out["n"])

    def col(s):
        return x[:, names.index(s)]

    assert col("CO")[0] < 1e-5 and col("H")[0] < 1e-7  # cold: essentially CO2 + 2 H2O + N2
    assert col("CO2")[0] == pytest.approx(1 / 10.52, rel=1e-3)  # 1 CO2 in 1 + 2 + 7.52 mol
    for s in ("CO2", "H2O"):
        assert (col(s).diff() < 0).all()  # break up as T rises
    for s in ("CO", "H2", "OH", "H", "O", "NO"):
        assert (col(s).diff() > 0).all()  # dissociation products grow
    assert col("CO")[-1] > col("CO2")[-1]  # by 3200 K, CO outnumbers CO2


def test_pressure_suppresses_dissociation():
    *_, low = _sweep([2600.0], P=1.0)
    *_, high = _sweep([2600.0], P=20.0)
    j = tensors.SPECIES.index("CO")
    assert equation.mole_fractions(high["n"])[0, j] < equation.mole_fractions(low["n"])[0, j]  # Le Chatelier


def test_adiabatic_flame_temperature_is_lowered_by_dissociation_and_peaks_slightly_rich():
    A, th = tensors.element_matrix(CPU), tensors.thermo(CPU)
    phis = torch.tensor([0.8, 0.9, 1.0, 1.05, 1.1, 1.2], dtype=F64)
    b, H = tensors.feed(phis), tensors.reactant_enthalpy(phis)
    T_ad, n_ad = solve.flame_temperature(A, b, H, 1.0, th)
    assert torch.allclose(equation.enthalpy(n_ad, T_ad, th), H, atol=1e-3)  # first law: H_products = H_reactants
    cold = solve.equilibrium(equation.standard_potential(torch.full_like(phis, 1000.0), th), A, b, 1.0)["n"]
    T_frozen = solve.frozen_flame_temperature(cold, H, th)
    assert (T_ad < T_frozen).all()
    assert 2150 < T_ad[2].item() < 2350  # accurate thermochemistry: about 2225 K at φ = 1
    assert phis[T_ad.argmax()].item() in (1.0, 1.05, 1.1)
