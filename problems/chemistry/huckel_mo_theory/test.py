"""Physics checks, written first (TDD): what any correct Hückel MO code must satisfy.

Run with `pytest problems/chemistry/huckel_mo_theory/test.py`. Each test is a physical law, not an implementation
detail: benzene and ring levels match α + 2β cos(2πk/N), alternant charges vanish, and 4n + 2 rings are aromatic.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _huckel():
    names, xys, A, pad, n_el = tensors.batch(CPU)
    return names, xys, solve.solve(A, pad, n_el, tensors.ALPHA, tensors.BETA)


def test_benzene_eigenvalues_and_delocalisation():
    alpha, beta = 0.3, -1.7  # any α and β < 0
    names, xys, A, pad, n_el = tensors.batch(CPU)
    res = solve.solve(A, pad, n_el, alpha, beta)
    b = names.index("benzene")
    E = res["E"][b, :6]
    expected = torch.tensor(
        [alpha + 2 * beta, alpha + beta, alpha + beta, alpha - beta, alpha - beta, alpha - 2 * beta]
    )
    assert torch.allclose(E, expected.double(), atol=1e-10)  # exact Hückel spectrum of a 6-ring
    assert math.isclose(res["E_pi"][b].item(), 6 * alpha + 8 * beta, abs_tol=1e-10)
    assert math.isclose(res["deloc"][b].item(), 2 * beta, abs_tol=1e-10)
    # cyclobutadiene: no delocalisation energy and a zero gap (half-filled degenerate pair)
    c = names.index("cyclobutadiene")
    assert abs(res["deloc"][c].item()) < 1e-10
    assert abs(res["gap"][c].item()) < 1e-10
    # padding atoms never take electrons
    assert res["n"].sum(-1).tolist() == [float(n) for n in n_el]  # electron count conserved


def test_butadiene_bond_orders_and_charges():
    names, xys, res = _huckel()
    b = names.index("butadiene")
    P = res["P"][b]
    bonds = [P[0, 1], P[1, 2], P[2, 3]]
    assert all(0 < p < 1 for p in bonds)
    assert math.isclose(bonds[0].item(), 2 / math.sqrt(5), rel_tol=1e-9)  # 0.894
    assert math.isclose(bonds[1].item(), 1 / math.sqrt(5), rel_tol=1e-9)  # 0.447
    assert math.isclose(bonds[2].item(), bonds[0].item(), rel_tol=1e-9)
    assert math.isclose(res["P"][names.index("benzene"), 0, 1].item(), 2 / 3, rel_tol=1e-9)
    # alternant hydrocarbons: every π charge vanishes
    for k, name in enumerate(names):
        assert res["q"][k, : len(xys[k])].abs().max() < 1e-10, name  # Coulson–Rushbrooke pairing theorem


def test_cyclic_eigenvalues_and_huckel_rule():
    for N in range(3, 15):
        E_num = torch.linalg.eigvalsh(-tensors.ring_adjacency(N))  # α = 0, β = -1
        E = equation.cyclic_levels(N, 0.0, -1.0)
        assert torch.allclose(E_num, E, atol=1e-12)
        # neutral rings with N π electrons: closed shell exactly for 4n + 2
        if N % 2 == 0:
            assert equation.is_closed_shell(E, N) == (N % 4 == 2), N  # Hückel's 4n + 2 rule
        # 4n + 2 electrons always close a shell (one lowest level, then degenerate pairs): the aromatic ions
        else:
            assert equation.is_closed_shell(E, tensors.ring_electrons(N)), N
            assert tensors.ring_electrons(N) % 4 == 2

    # benzene is stabilised against hexatriene, cyclobutadiene destabilised against butadiene
    def ring_minus_chain(N):
        ring, chain = equation.cyclic_levels(N, 0.0, -1.0), equation.chain_levels(N, 0.0, -1.0)
        e = [equation.pi_energy(L[None], equation.occupations(L[None], [N]))[0] for L in (ring, chain)]
        return (e[0] - e[1]).item()

    assert ring_minus_chain(6) < -1.0  # aromatic
    assert ring_minus_chain(4) > 0.4  # antiaromatic
