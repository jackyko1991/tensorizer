"""Physics checks, written first (TDD): what any correct phylogenetics code must satisfy.

Run with `pytest problems/bioinformatics/phylogenetic_tree/test.py`. Each test is an exact invariant: the
Jukes–Cantor matrix is a Markov semigroup, corrected distances are additive on the true tree, and neighbour joining
and UPGMA recover the topology.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_jukes_cantor_matrix_is_a_markov_semigroup():
    P = equation.jc_transition
    assert torch.allclose(P(0.4, CPU).sum(1), torch.ones(4))
    assert torch.allclose(P(0.3, CPU) @ P(0.5, CPU), P(0.8, CPU), atol=1e-6)
    assert torch.allclose(P(0.0, CPU), torch.eye(4))


def test_p_distance_einsum_counts_mismatches():
    X = tensors.simulate(tensors.random_tree(5, 0.5, True, 1), 300, 1, 2, CPU)[0]
    letters = X.argmax(-1)
    direct = (letters[:, None, :] != letters[None, :, :]).float().mean(-1)
    assert torch.allclose(equation.p_distance(X), direct, atol=1e-6)


def test_jukes_cantor_distances_are_additive_on_the_true_tree_in_expectation():
    tree = tensors.random_tree(6, 0.35, False, seed=4)
    M, b = tensors.path_matrix(tree, CPU)
    d_true = equation.patristic(M, b)
    X = tensors.simulate(tree, 400, 200, seed=6, device=CPU)  # (R, n, L, 4): 200 replicate alignments
    p = equation.p_distance(X)  # (R, n, n)
    d_mean = equation.jukes_cantor(p).mean(0)
    off = ~torch.eye(6, dtype=torch.bool)
    rel = ((d_mean - d_true).abs() / d_true)[off]
    assert rel.max() < 0.05
    far = d_true == d_true.max()
    assert (p.mean(0)[far] < 0.9 * d_true[far]).all()  # the raw p-distance underestimates long paths


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_neighbour_joining_recovers_topology_from_additive_distances(seed):
    tree = tensors.random_tree(12, 0.6, False, seed)  # no clock: not ultrametric
    D = equation.patristic(*tensors.path_matrix(tree, CPU)).double()
    nj = solve.neighbour_joining(D)
    assert solve.robinson_foulds(nj, tree) == 0
    assert torch.allclose(equation.patristic(*tensors.path_matrix(nj, CPU)).double(), D, atol=1e-5)
    rooted = solve.midpoint_root(nj)
    assert solve.splits(rooted) == solve.splits(nj)


def test_upgma_recovers_clock_trees_and_rf_counts_differences():
    tree = tensors.random_tree(10, 0.5, True, seed=3)
    D = equation.patristic(*tensors.path_matrix(tree, CPU))
    up = solve.upgma(D)
    assert solve.robinson_foulds(up, tree) == 0
    other = tensors.random_tree(10, 0.5, True, seed=8)
    rf = solve.robinson_foulds(other, tree)
    assert rf % 2 == 0 and 0 < rf <= 2 * (10 - 3)
