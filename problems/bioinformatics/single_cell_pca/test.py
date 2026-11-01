"""Physics checks, written first (TDD): what any correct single-cell code must satisfy.

Run with `pytest problems/bioinformatics/single_cell_pca/test.py`. Each test is an exact invariant: counts are
negative-binomial (variance m + φ m²), normalisation removes library size, PCA variance ratios are sorted and sum to
one, and k-means never increases its inertia.
"""

import itertools

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def cells():
    mu, markers = tensors.type_profiles(800, 4, 40, 4.0, 0, CPU)
    counts, labels, _ = tensors.sample_counts(600, mu, 0.3, 0.5, 1)
    return counts, labels, markers


def test_negative_binomial_counts_are_overdispersed():
    mu = torch.full((1, 50), 5.0)
    counts, _, _ = tensors.sample_counts(4000, mu, 0.5, 0.0, 3)
    m, v = counts.mean(), counts.var(0).mean()
    assert m.item() == pytest.approx(5.0, rel=0.05)
    assert v.item() == pytest.approx(5.0 + 0.5 * 25.0, rel=0.1)  # m + φ m²


def test_normalisation_removes_library_size(cells):
    counts, _, _ = cells
    scaled, X = equation.normalise(counts)
    assert torch.allclose(scaled.sum(1), torch.full((len(counts),), 1e4))
    assert torch.allclose(X, torch.log1p(scaled))


def test_pca_explained_variance_is_sorted_and_sums_to_at_most_one(cells):
    counts, _, _ = cells
    scaled, X = equation.normalise(counts)
    hvg = torch.argsort(equation.hvg_scores(scaled), descending=True)[:150]
    y, V, ratio = equation.pca(X[:, hvg], 10)
    assert (ratio[1:] <= ratio[:-1] + 1e-7).all()
    assert ratio[:10].sum() <= 1.0
    assert ratio.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(V.T @ V, torch.eye(10), atol=1e-4)  # orthonormal loadings
    # the score variance along each PC is its share of the total variance
    Z = X[:, hvg] - X[:, hvg].mean(0)
    assert torch.allclose(y.var(0) / Z.var(0).sum(), ratio[:10], rtol=1e-3)


def test_hvgs_are_enriched_in_marker_genes(cells):
    counts, _, markers = cells
    scaled, _ = equation.normalise(counts)
    hvg = torch.argsort(equation.hvg_scores(scaled), descending=True)[:100]
    hit = torch.isin(hvg, markers.flatten()).float().mean().item()
    assert hit > 3 * markers.numel() / counts.shape[1]  # far above the 20% base rate


def test_kmeans_never_increases_the_inertia(cells):
    counts, _, _ = cells
    _, X = equation.normalise(counts)
    y, _, _ = equation.pca(X, 8)
    km = solve.kmeans(y, 5, 6, 15, 0)
    inertia = km["inertia"]  # (T+1, R)
    assert (inertia[1:] <= inertia[:-1] * (1 + 1e-6) + 1e-6).all()
    assert km["labels"].shape == (16, y.shape[0])


def test_ari_is_one_for_relabelled_partitions_and_near_zero_for_random():
    gen = torch.Generator().manual_seed(4)
    a = torch.randint(4, (500,), generator=gen)
    perm = torch.tensor([2, 0, 3, 1])
    assert equation.adjusted_rand_index(a, perm[a]) == pytest.approx(1.0)
    b = torch.randint(4, (500,), generator=gen)
    assert abs(equation.adjusted_rand_index(a, b)) < 0.05


def test_ari_by_brute_force_pair_counting():
    a = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0])
    b = torch.tensor([1, 1, 0, 0, 0, 2, 2, 2])
    same_a = [a[i] == a[j] for i, j in itertools.combinations(range(8), 2)]
    same_b = [b[i] == b[j] for i, j in itertools.combinations(range(8), 2)]
    n11 = sum(bool(x and y) for x, y in zip(same_a, same_b))
    na, nb, n = sum(map(bool, same_a)), sum(map(bool, same_b)), len(same_a)
    expected = (n11 - na * nb / n) / (0.5 * (na + nb) - na * nb / n)
    assert equation.adjusted_rand_index(a, b) == pytest.approx(expected)


def test_pipeline_ari_is_high_on_well_separated_types(cells):
    counts, labels, _ = cells
    scaled, X = equation.normalise(counts)
    hvg = torch.argsort(equation.hvg_scores(scaled), descending=True)[:200]
    y, _, _ = equation.pca(X[:, hvg], 10)
    km = solve.kmeans(y, 4, 8, 15, 0)
    assert equation.adjusted_rand_index(km["labels"][-1], labels) > 0.9
