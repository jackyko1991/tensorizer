"""Physics checks, written first (TDD): what any correct coevolution code must satisfy.

Run with `pytest problems/bioinformatics/coevolution_contacts/test.py`. Each test is an exact invariant: pair
frequencies marginalise to single-site ones, APC removes row and column means, mean-field couplings invert the
covariance, and DCA finds planted contacts far above chance.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_lattice_fold_is_a_chain_and_contacts_join_opposite_colours():
    coords = tensors.lattice_fold((3, 3, 4), CPU)
    assert len({tuple(c) for c in coords.tolist()}) == 36  # self-avoiding: every site once
    assert ((coords[1:] - coords[:-1]).abs().sum(-1) == 1).all()  # consecutive residues are neighbours
    contacts = tensors.contact_map(coords)
    assert torch.equal(contacts, contacts.T)
    colour = tensors.parity(coords)
    i, j = torch.nonzero(contacts, as_tuple=True)
    assert (colour[i] != colour[j]).all()  # what makes checkerboard Gibbs sampling exact


def test_pair_frequencies_are_consistent_with_single_site_marginals():
    gen = torch.Generator().manual_seed(0)
    L, q = 7, 4
    x = tensors.one_hot(torch.randint(q, (300, L), generator=gen), q)
    f1, f2 = equation.single_frequencies(x), equation.pair_frequencies(x)
    assert torch.allclose(f2.sum(3), f1[:, :, None].expand(L, q, L), atol=1e-6)  # Σ_b f_iajb = f_ia
    assert torch.allclose(f2.sum(1), f1[None, :, :].expand(L, L, q), atol=1e-6)  # Σ_a f_iajb = f_jb
    for i in range(L):
        assert torch.allclose(f2[i, :, i, :], torch.diag(f1[i]), atol=1e-6)  # f_iaib = f_ia δ_ab
    # the pseudocount keeps the same consistency
    g1, g2 = equation.with_pseudocount(f1, f2, 0.3)
    assert torch.allclose(g2.sum(3), g1[:, :, None].expand(L, q, L), atol=1e-6)
    assert torch.allclose(g1.sum(1), torch.ones(L))


def test_apc_removes_row_and_column_means():
    gen = torch.Generator().manual_seed(1)
    A = torch.rand(20, 20, generator=gen, dtype=torch.float64)
    F = (A + A.T).fill_diagonal_(0.0)
    G = equation.apc(F)
    assert G.mean(1).abs().max() < 1e-12
    assert G.mean(0).abs().max() < 1e-12
    assert torch.allclose(G, G.T)


def test_mean_field_couplings_recover_a_gaussian_precision_matrix_for_q2():
    # For q = 2 one state per column remains; then J = −C⁻¹ is exactly minus the inverse covariance.
    gen = torch.Generator().manual_seed(2)
    s = torch.randint(2, (2000, 5), generator=gen)
    x = tensors.one_hot(s, 2)
    C = equation.covariance(equation.single_frequencies(x), equation.pair_frequencies(x))
    J = equation.mean_field_couplings(C)
    expected = -torch.linalg.inv(C[:, 0, :, 0])
    off = ~torch.eye(5, dtype=torch.bool)
    assert torch.allclose(J[:, 0, :, 0][off], expected[off], atol=1e-4)


@pytest.fixture(scope="module")
def planted_msa():
    coords = tensors.lattice_fold((3, 3, 4), CPU)
    contacts = tensors.contact_map(coords)
    h, J = tensors.planted_potts(contacts, 5, 1.5, 1.0, 0)
    seqs = solve.gibbs_sample(h, J, tensors.parity(coords), 800, 100, 1)
    return tensors.one_hot(seqs, 5), contacts


def test_dca_precision_is_far_above_random(planted_msa):
    x, contacts = planted_msa
    L = contacts.shape[0]
    n_pairs = (L - 3) * (L - 2) // 2
    random_level = contacts.sum().item() / 2 / n_pairs
    p_dca = equation.precision(equation.top_pairs(solve.mfdca_scores(x, 0.3), L), contacts)
    p_cov = equation.precision(equation.top_pairs(solve.covariance_scores(x, 0.3), L), contacts)
    assert random_level < 0.1
    assert p_dca > 0.75
    assert p_dca >= p_cov  # removing indirect correlations helps


def test_pseudo_likelihood_also_finds_the_contacts(planted_msa):
    x, contacts = planted_msa
    L = contacts.shape[0]
    scores, _, _, losses = solve.train_plm(x, 150, 0.05, 1e-4, n_snapshots=3)
    assert losses[-1] < losses[0]
    assert equation.precision(equation.top_pairs(scores, L), contacts) > 0.7
