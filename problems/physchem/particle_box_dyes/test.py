"""Physics checks, written first (TDD): what any correct particle-in-a-box dye code must satisfy.

Run with `pytest problems/physchem/particle_box_dyes/test.py`. Each test is a physical law, not an implementation
detail: levels scale as n², eigenstates are normalised, longer chains absorb redder, and bond alternation opens a gap.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def _box(series="cyanine", n_max=4, V0=0.0, M=300):
    names, n_bonds, n_el = tensors.molecules(series, n_max, CPU)
    L = tensors.box_length(n_bonds, 0.139, 1.0).to(F64)
    x, dx = tensors.grids(L, M)
    V = equation.alternation_potential(x, V0, 0.139, 1.0)
    E, psi = solve.solve(equation.hamiltonian(dx, V), dx, int(n_el.max()) // 2 + 2)
    return L, n_el, E, psi, dx


def test_numerical_box_levels_scale_as_n_squared():
    L, _, E, psi, dx = _box()
    n = torch.arange(1, E.shape[1] + 1, dtype=F64)
    assert torch.allclose(E / E[:, :1], (n**2).expand_as(E), rtol=2e-3)  # E_n = n² h² / (8 m L²)
    exact = equation.energy_levels(n, L)
    assert ((E - exact) / exact).abs().max() < 2e-3
    norm = (psi**2).sum(1) * dx[:, None]  # Σ ψ² dx = 1 for every state
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-10)  # probability is normalised


def test_box_error_shrinks_with_grid():
    errs = []
    for M in (100, 200):
        L, _, E, _, _ = _box(n_max=1, M=M)
        exact = equation.energy_levels(torch.arange(1, E.shape[1] + 1, dtype=F64), L)
        errs.append(((E - exact) / exact).abs().max().item())
    assert 3.5 < errs[0] / errs[1] < 4.5  # second-order finite differences


def test_gap_formula_matches_levels_and_lambda_grows():
    L, n_el, E, _, _ = _box()
    n = torch.arange(1, E.shape[1] + 1, dtype=F64)
    exact = equation.energy_levels(n, L)
    homo = (n_el // 2 - 1)[:, None]
    gap_levels = (exact.gather(1, homo + 1) - exact.gather(1, homo))[:, 0]
    gap = equation.homo_lumo_gap(n_el, L)
    assert torch.allclose(gap, gap_levels, rtol=1e-6)
    lam = equation.wavelength(gap)
    assert (lam[1:] > lam[:-1]).all()  # longer chain → redder absorption
    # cyanine with 3 CH groups (k = 1): L = 6 l, N = 6  →  λ = 8 m c (6 l)² / (7 h) ≈ 328 nm
    assert abs(lam[1].item() - 327.6) < 1.0


def test_alternation_opens_a_gap():
    _, n_el, E0, _, _ = _box("polyene", 6, V0=0.0)
    _, _, E1, _, _ = _box("polyene", 6, V0=1.5)
    homo = (n_el // 2 - 1)[:, None]
    g0 = (E0.gather(1, homo + 1) - E0.gather(1, homo))[:, 0]
    g1 = (E1.gather(1, homo + 1) - E1.gather(1, homo))[:, 0]
    assert (g1 > g0).all()
    assert g1[-1] > 0.6 * 1.5  # the gap stays of order V0 for long chains, while the plain box gap → 0
