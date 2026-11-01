"""Physics checks, written first (TDD): what any correct chemical-equilibrium code must satisfy.

Run with `pytest problems/physchem/chemical_equilibrium/test.py`. Each test is a physical law, not an implementation
detail: Q = K at equilibrium, Le Chatelier's principle for pressure, temperature and added product, and van 't Hoff.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def _equilibrium(name, T, P, n0=None):
    r = tensors.reaction(name, CPU)
    T = torch.as_tensor(T, dtype=F64).reshape(-1)
    P = torch.as_tensor(P, dtype=F64).reshape(-1)
    T, P = torch.broadcast_tensors(T, P)
    n0 = (r["n0"] if n0 is None else torch.as_tensor(n0, dtype=F64)).expand(len(T), -1)
    lnK = equation.log_K(T, r["dH"], r["dS"])
    xi, _ = solve.equilibrium_extent(n0, r["nu"], P, lnK)
    n = equation.moles(n0, r["nu"], xi)
    return r, n, xi, lnK, P


@pytest.mark.parametrize("name", ["n2o4", "haber"])
def test_equilibrium_satisfies_Q_equals_K(name):
    r = tensors.reaction(name, CPU)
    T_b, P_b = tensors.condition_grid(r["T_curves"], tensors.pressures(CPU, 13))
    _, n, _, lnK, P = _equilibrium(name, T_b, P_b)
    assert (equation.log_Q(n, P, r["nu"]) - lnK).abs().max() < 1e-10  # law of mass action: Q = K
    assert n.min() > 0  # amounts stay positive


def test_n2o4_matches_hand_formula():
    """K = 4α²P/((1−α²)P°)  ⇒  α = √(K/(K + 4P))."""
    T = torch.tensor([280.0, 320.0, 360.0], dtype=F64)
    for P in (0.1, 1.0, 30.0):
        r, _, xi, lnK, _ = _equilibrium("n2o4", T, P)
        K = lnK.exp()
        assert torch.allclose(xi, torch.sqrt(K / (K + 4 * P)), rtol=1e-10)  # analytic degree of dissociation


def test_k298_n2o4_textbook_value():
    r = tensors.reaction("n2o4", CPU)
    K = math.exp(equation.log_K(torch.tensor(298.15, dtype=F64), r["dH"], r["dS"]).item())
    assert 0.1 < K < 0.2  # textbook K_p ≈ 0.15 at 25 °C


def test_le_chatelier_pressure():
    P = torch.logspace(-1, 2, 7, dtype=F64)
    _, _, xi_n2o4, _, _ = _equilibrium("n2o4", 320.0, P)
    _, _, xi_nh3, _, _ = _equilibrium("haber", 700.0, P)
    assert (xi_n2o4.diff() < 0).all()  # Δν = +1: pressure pushes back to N2O4
    assert (xi_nh3.diff() > 0).all()  # Δν = −2: pressure makes more NH3


def test_le_chatelier_temperature():
    T = torch.linspace(280.0, 380.0, 6, dtype=F64)
    _, _, xi_n2o4, _, _ = _equilibrium("n2o4", T, 1.0)
    _, _, xi_nh3, _, _ = _equilibrium("haber", T + 400.0, 100.0)
    assert (xi_n2o4.diff() > 0).all()  # endothermic: heating favours products
    assert (xi_nh3.diff() < 0).all()  # exothermic: heating favours reactants


@pytest.mark.parametrize("name", ["n2o4", "haber"])
def test_le_chatelier_added_product(name):
    T = 320.0 if name == "n2o4" else 700.0
    r, n_eq, _, lnK, P = _equilibrium(name, T, 5.0)
    n_add, gap, n_new = solve.add_product_and_reequilibrate(n_eq, r["nu"], P, lnK, 0.3)
    assert gap.item() > 0  # Q > K right after adding product
    assert n_new[0, -1] < n_add[0, -1]  # some product turns back
    assert n_new[0, -1] > n_eq[0, -1]  # ... but not all of it
    assert (n_new[0, :-1] > n_eq[0, :-1]).all()  # reactants increase
    assert abs(equation.log_Q(n_new, P, r["nu"]) - lnK).item() < 1e-10


@pytest.mark.parametrize("name", ["n2o4", "haber"])
def test_van_t_hoff(name):
    r = tensors.reaction(name, CPU)
    inv_T = (1.0 / torch.linspace(300.0, 800.0, 11, dtype=F64)).requires_grad_(True)
    lnK = equation.log_K(1.0 / inv_T, r["dH"], r["dS"])
    (slope,) = torch.autograd.grad(lnK.sum(), inv_T)
    assert torch.allclose(slope, torch.full_like(slope, equation.van_t_hoff_slope(r["dH"])), rtol=1e-12)
    # integrated form: ln(K2/K1) = −ΔH°/R (1/T2 − 1/T1)
    T1, T2 = torch.tensor(350.0, dtype=F64), torch.tensor(450.0, dtype=F64)
    lhs = equation.log_K(T2, r["dH"], r["dS"]) - equation.log_K(T1, r["dH"], r["dS"])
    assert math.isclose(lhs.item(), -r["dH"] / equation.R_GAS * (1 / 450.0 - 1 / 350.0), rel_tol=1e-12)
    # ΔG° = −RT ln K
    dG = equation.standard_gibbs(T1, r["dH"], r["dS"]).item()
    assert math.isclose(dG, -equation.R_GAS * 350.0 * equation.log_K(T1, r["dH"], r["dS"]).item(), rel_tol=1e-12)
