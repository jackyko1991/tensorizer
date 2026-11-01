"""Physics checks, written first (TDD): what any correct Hartree–Fock H₂ code must satisfy.

Run with `pytest problems/physchem/hartree_fock_h2/test.py`. Each test is a physical law, not an implementation
detail: the overlap is normalised and positive definite, SCF is self-consistent with 2 electrons, full CI dissociates.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _hf(name="H2", R=(1.4,)):
    Z, zeta, n_atom = tensors.molecule(name, CPU)
    alpha, coef = tensors.sto3g(zeta)
    nuclei = tensors.geometry(torch.tensor(R, dtype=torch.float64))
    ints = solve.integrals(alpha, coef, nuclei, Z)
    res = solve.scf(ints, tensors.N_ELECTRONS)
    E = res["E_el"] + equation.nuclear_repulsion(nuclei, Z)
    return ints, res, E, (alpha, coef, Z, n_atom)


def test_overlap_normalised_and_positive_definite():
    ints, *_ = _hf(R=(0.7, 1.4, 3.0))
    S = ints["S"]
    assert torch.allclose(S.diagonal(dim1=-2, dim2=-1), torch.ones(3, 2, dtype=S.dtype), atol=1e-5)  # ⟨φ|φ⟩ = 1
    assert torch.allclose(S, S.transpose(-1, -2))
    assert torch.linalg.eigvalsh(S).min() > 0  # a Gram matrix of independent functions is positive definite
    assert abs(S[1, 0, 1].item() - 0.6593) < 1e-3  # Szabo & Ostlund S₁₂ at R = 1.4 bohr


def test_h2_energy_matches_szabo_ostlund():
    ints, res, E, _ = _hf()
    assert abs(E[0].item() - (-1.1167)) < 1e-3
    assert abs(res["E_el"][0].item() - (-1.8310)) < 1e-3


def test_fock_symmetric_converged_and_two_electrons():
    for name, R in (("H2", 1.4), ("HeH+", 1.4632)):
        ints, res, E, _ = _hf(name, (R, 2.5))
        F, P, S = res["F"], res["P"], ints["S"]
        assert res["converged"]
        assert torch.allclose(F, F.transpose(-1, -2), atol=1e-12)  # Fock operator is Hermitian
        n = torch.einsum("bmn,bnm->b", P, S)
        assert torch.allclose(n, torch.full_like(n, 2.0), atol=1e-8)  # electron number: tr(PS) = 2
        # self-consistency: P built from the eigenvectors of F(P) reproduces F
        assert torch.allclose(equation.fock(ints["H"], P, ints["eri"]), F, atol=1e-10)
    _, _, E, _ = _hf("HeH+", (1.4632,))
    assert abs(E[0].item() - (-2.8607)) < 1e-3


def test_dissociation_failure_and_full_ci_limit():
    ints, res, E, (alpha, coef, Z, n_atom) = _hf(R=(1.4, 10.0))
    E_nuc = equation.nuclear_repulsion(tensors.geometry(torch.tensor([1.4, 10.0], dtype=torch.float64)), Z)
    E_ci = solve.full_ci(ints, res["C"]) + E_nuc
    limit = solve.separated_atoms(alpha, coef, Z, n_atom)
    assert abs(limit - 2 * (-0.4666)) < 1e-3  # two STO-3G hydrogen atoms
    assert E[1] - limit > 0.2  # RHF stays far above H + H
    assert abs(E_ci[1].item() - limit) < 1e-3  # full CI dissociates correctly (size consistency)
    assert -0.03 < (E_ci[0] - E[0]).item() < -0.01  # variational: correlation energy ≈ -0.02 hartree, below HF
