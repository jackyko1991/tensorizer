"""Physics checks, written first (TDD): what any correct limiting-reagent and yield code must satisfy.

Run with `pytest problems/chemistry/limiting_reagent_yield/test.py`. Each test is a physical law, not an implementation
detail: the limiting reagent matches a hand calculation, mass is conserved at every extent, and no amount goes negative.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_molar_masses():
    _, molar, _ = tensors.stoichiometry("aluminium_iodine", CPU)
    assert torch.allclose(molar, torch.tensor([26.982, 253.808, 407.694], dtype=torch.float64))
    _, molar, _ = tensors.stoichiometry("haber", CPU)
    assert torch.allclose(molar, torch.tensor([28.014, 2.016, 17.031], dtype=torch.float64))


def test_limiting_reagent_against_hand_calculation():
    nu, molar, _ = tensors.stoichiometry("aluminium_iodine", CPU)
    m0, weighed = tensors.experiments("aluminium_iodine", 1.20, 2.40, 2.00, CPU)
    res = solve.analyse(m0, weighed, nu, molar)
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
    nu, molar, _ = tensors.stoichiometry("haber", CPU)
    m0 = torch.tensor([[28.0, 10.0, 0.0], [28.0, 3.0, 0.0]], dtype=torch.float64)
    res = solve.analyse(m0, torch.tensor([5.0, 5.0], dtype=torch.float64), nu, molar)
    # 1 mol N2 with 4.96 mol H2: N2 limiting, 2 mol NH3;  with 1.49 mol H2: H2 limiting, 0.992 mol NH3
    assert res["limiting"].tolist() == [0, 1]
    assert math.isclose(res["theory"][0].item(), 2 * 28.0 / 28.014 * 17.031, rel_tol=1e-12)
    assert math.isclose(res["theory"][1].item(), 2 * (3.0 / 2.016 / 3) * 17.031, rel_tol=1e-12)
    # total mass is conserved at every extent (atomic masses add up exactly)
    xi, n = solve.extent_sweep(res["n0"], nu, res["xi_max"])
    mass = torch.einsum("bks,s->bk", n, molar)
    assert torch.allclose(mass, m0.sum(-1, keepdim=True).expand_as(mass), rtol=1e-12)  # Lavoisier: mass conserved
    assert (n >= -1e-12).all()  # no amount goes negative


def test_yield_map_matches_batch_and_stoichiometric_line():
    nu, molar, _ = tensors.stoichiometry("aluminium_iodine", CPU)
    g1 = torch.linspace(0.0, 2.0, 21, dtype=torch.float64)
    g2 = torch.linspace(0.0, 28.0, 15, dtype=torch.float64)
    pm = solve.yield_map(g1, g2, nu, molar)
    assert pm.shape == (15, 21)
    xi, _ = equation.max_extent(torch.stack([g1 / molar[0], torch.full_like(g1, 14.0) / molar[1], 0 * g1], -1), nu)
    assert torch.allclose(pm[7], equation.theoretical_mass(xi, nu, molar))
    # on the stoichiometric line both reactants run out together: product mass = m_Al + m_I2
    ratio = (3 * molar[1]) / (2 * molar[0])
    m_al = torch.tensor([0.5, 1.0], dtype=torch.float64)
    n0 = torch.stack([m_al / molar[0], ratio * m_al / molar[1], torch.zeros(2, dtype=torch.float64)], -1)
    xi, _ = equation.max_extent(n0, nu)
    assert torch.allclose(equation.theoretical_mass(xi, nu, molar), m_al * (1 + ratio), rtol=1e-10)
