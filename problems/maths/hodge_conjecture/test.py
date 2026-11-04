"""Mathematical checks, written first (TDD): what any correct Hodge-number code must satisfy.

Run with `pytest problems/maths/hodge_conjecture/test.py`. Each test is a theorem: Hodge symmetry and Serre duality,
the Euler characteristic of a degree-d hypersurface, the classical Hodge numbers (K3, quintic threefold, cubic
fourfold), Shioda's Picard numbers of Fermat surfaces, the Hodge index theorem on the lattice spanned by lines, and
the Künneth formula.
"""

import math

import pytest
import torch

from . import equation, tensors


def _diamonds():
    out = [equation.fermat_diamond(n, d) for n in range(1, 5) for d in range(2, 8)]
    out += [equation.torus_diamond(g) for g in range(1, 5)]
    k3, e = equation.fermat_diamond(2, 4), equation.torus_diamond(1)
    out += [equation.kunneth(k3, e), equation.kunneth(k3, k3), equation.kunneth(equation.fermat_diamond(1, 4), e)]
    return out


def test_hodge_symmetry_and_serre_duality():
    for h in _diamonds():
        assert torch.equal(h, h.T)  # h^{p,q} = h^{q,p} (complex conjugation)
        assert torch.equal(h, h.flip(0).flip(1))  # h^{p,q} = h^{n−p,n−q} (Serre duality)
        assert (h >= 0).all() and h[0, 0] == 1  # connected


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_euler_characteristic_of_fermat_hypersurfaces(n):
    for d in range(1, 9):
        h = equation.fermat_diamond(n, d)
        assert equation.euler_characteristic(h) == equation.euler_formula(n, d)  # ((1−d)^{n+2} − 1)/d + n + 2


def test_classical_hodge_numbers():
    for d in range(1, 9):
        assert equation.fermat_diamond(1, d)[1, 0] == (d - 1) * (d - 2) // 2  # genus of a plane curve
    k3 = equation.fermat_diamond(2, 4)
    assert k3.tolist() == [[1, 0, 1], [0, 20, 0], [1, 0, 1]]
    assert equation.fermat_diamond(2, 3)[1, 1] == 7  # cubic surface: P² blown up in 6 points
    quintic = equation.fermat_diamond(3, 5)
    assert (quintic[3, 0], quintic[2, 1], quintic[1, 1]) == (1, 101, 1)  # the quintic Calabi–Yau threefold
    cubic4 = equation.fermat_diamond(4, 3)
    assert (cubic4[4, 0], cubic4[3, 1], cubic4[2, 2]) == (0, 1, 21)  # cubic fourfold: a K3-like middle


def test_jacobian_ring_count_equals_character_count():
    """Griffiths: H^{n−q,q}_prim ≅ Jacobian ring in degree (q+1)d − (n+2); Shioda: characters a with Σ⟨a_i/d⟩ = q+1."""
    for n, d in ((1, 5), (2, 4), (2, 6), (3, 5), (4, 4)):
        jac = equation.jacobian_hodge(n, d)
        chars = tensors.characters(n, d)
        q = chars.sum(dim=1) // d - 1
        assert torch.equal(torch.bincount(q, minlength=n + 1)[: n + 1], jac)


def test_picard_numbers_of_fermat_surfaces():
    known = {3: 7, 4: 20, 5: 37, 6: 86, 7: 91}
    for d, rho in known.items():
        assert equation.hodge_class_count(2, d) == rho
        assert rho <= equation.fermat_diamond(2, d)[1, 1]  # Hodge classes ⊂ H^{1,1}
    for d in (5, 7, 11, 13):  # Shioda: ρ = 3(d − 1)(d − 2) + 1 when gcd(d, 6) = 1
        assert equation.hodge_class_count(2, d) == 3 * (d - 1) * (d - 2) + 1


def test_lines_span_the_neron_severi_lattice_with_hodge_index_signature():
    for d, n_lines, rho in ((3, 27, 7), (4, 48, 20)):
        gram = equation.line_gram(d)
        assert gram.shape == (n_lines, n_lines) and torch.equal(gram, gram.T)
        assert (gram.diagonal() == 2 - d).all()  # adjunction: L² = 2 − d
        ev = torch.linalg.eigvalsh(gram.double())
        tol = 1e-8 * ev.abs().max()
        assert int((ev.abs() > tol).sum()) == rho  # the lines span all Hodge classes (Lefschetz (1,1))
        assert int((ev > tol).sum()) == 1  # Hodge index theorem: signature (1, ρ − 1)
    assert ((equation.line_gram(3) == 1).sum(dim=1) == 10).all()  # each of the 27 lines meets exactly 10 others


def test_every_middle_class_of_the_fermat_cubic_fourfold_is_hodge():
    assert equation.hodge_class_count(4, 3) == equation.fermat_diamond(4, 3)[2, 2] == 21


def test_complex_tori_and_kunneth():
    for g in range(1, 6):
        h = equation.torus_diamond(g)
        assert h[1, 0] == g and h[g, g] == 1
        assert int(h.sum()) == 4**g  # total Betti number of (S¹)^{2g}
        assert equation.euler_characteristic(h) == 0
        for p in range(g + 1):
            assert h[p, g - p] == math.comb(g, p) ** 2
    k3, quintic = equation.fermat_diamond(2, 4), equation.fermat_diamond(3, 5)
    prod = equation.kunneth(k3, quintic)
    assert equation.euler_characteristic(prod) == 24 * -200  # χ is multiplicative


def test_lines_span_neron_severi_exactly_when_d_le_4_or_coprime_to_6():
    """Shioda: over ℚ the 3d² lines span NS of the Fermat surface iff d ≤ 4 or gcd(d, 6) = 1 (over ℤ too for
    gcd(d, 6) = 1, d < 100: Schütt–Shioda–van Luijk 2010)."""
    for d in range(3, 10):
        ev = torch.linalg.eigvalsh(equation.line_gram(d).double())
        rank = int((ev.abs() > 1e-8 * ev.abs().max()).sum())
        rho = equation.hodge_class_count(2, d)
        assert (rank == rho) == (d <= 4 or math.gcd(d, 6) == 1)
        assert rank <= rho
