"""Checks for the advanced bioinformatics problems: coevolution contacts, single-cell PCA, the DNA-binding CNN."""

import itertools

import pytest
import torch

from problems.bioinformatics.coevolution_contacts import equation as co_eq
from problems.bioinformatics.coevolution_contacts import solve as co_solve
from problems.bioinformatics.coevolution_contacts import tensors as co_t
from problems.bioinformatics.dna_binding_cnn import equation as dna_eq
from problems.bioinformatics.dna_binding_cnn import solve as dna_solve
from problems.bioinformatics.dna_binding_cnn import tensors as dna_t
from problems.bioinformatics.single_cell_pca import equation as sc_eq
from problems.bioinformatics.single_cell_pca import solve as sc_solve
from problems.bioinformatics.single_cell_pca import tensors as sc_t

CPU = torch.device("cpu")

# --- coevolution contacts ---


def test_lattice_fold_is_a_chain_and_contacts_join_opposite_colours():
    coords = co_t.lattice_fold((3, 3, 4), CPU)
    assert len({tuple(c) for c in coords.tolist()}) == 36  # self-avoiding: every site once
    assert ((coords[1:] - coords[:-1]).abs().sum(-1) == 1).all()  # consecutive residues are neighbours
    contacts = co_t.contact_map(coords)
    assert torch.equal(contacts, contacts.T)
    colour = co_t.parity(coords)
    i, j = torch.nonzero(contacts, as_tuple=True)
    assert (colour[i] != colour[j]).all()  # what makes checkerboard Gibbs sampling exact


def test_pair_frequencies_are_consistent_with_single_site_marginals():
    gen = torch.Generator().manual_seed(0)
    L, q = 7, 4
    x = co_t.one_hot(torch.randint(q, (300, L), generator=gen), q)
    f1, f2 = co_eq.single_frequencies(x), co_eq.pair_frequencies(x)
    assert torch.allclose(f2.sum(3), f1[:, :, None].expand(L, q, L), atol=1e-6)  # Σ_b f_iajb = f_ia
    assert torch.allclose(f2.sum(1), f1[None, :, :].expand(L, L, q), atol=1e-6)  # Σ_a f_iajb = f_jb
    for i in range(L):
        assert torch.allclose(f2[i, :, i, :], torch.diag(f1[i]), atol=1e-6)  # f_iaib = f_ia δ_ab
    # the pseudocount keeps the same consistency
    g1, g2 = co_eq.with_pseudocount(f1, f2, 0.3)
    assert torch.allclose(g2.sum(3), g1[:, :, None].expand(L, q, L), atol=1e-6)
    assert torch.allclose(g1.sum(1), torch.ones(L))


def test_apc_removes_row_and_column_means():
    gen = torch.Generator().manual_seed(1)
    A = torch.rand(20, 20, generator=gen, dtype=torch.float64)
    F = (A + A.T).fill_diagonal_(0.0)
    G = co_eq.apc(F)
    assert G.mean(1).abs().max() < 1e-12
    assert G.mean(0).abs().max() < 1e-12
    assert torch.allclose(G, G.T)


def test_mean_field_couplings_recover_a_gaussian_precision_matrix_for_q2():
    # For q = 2 one state per column remains; then J = −C⁻¹ is exactly minus the inverse covariance.
    gen = torch.Generator().manual_seed(2)
    s = torch.randint(2, (2000, 5), generator=gen)
    x = co_t.one_hot(s, 2)
    C = co_eq.covariance(co_eq.single_frequencies(x), co_eq.pair_frequencies(x))
    J = co_eq.mean_field_couplings(C)
    expected = -torch.linalg.inv(C[:, 0, :, 0])
    off = ~torch.eye(5, dtype=torch.bool)
    assert torch.allclose(J[:, 0, :, 0][off], expected[off], atol=1e-4)


@pytest.fixture(scope="module")
def planted_msa():
    coords = co_t.lattice_fold((3, 3, 4), CPU)
    contacts = co_t.contact_map(coords)
    h, J = co_t.planted_potts(contacts, 5, 1.5, 1.0, 0)
    seqs = co_solve.gibbs_sample(h, J, co_t.parity(coords), 800, 100, 1)
    return co_t.one_hot(seqs, 5), contacts


def test_dca_precision_is_far_above_random(planted_msa):
    x, contacts = planted_msa
    L = contacts.shape[0]
    n_pairs = (L - 3) * (L - 2) // 2
    random_level = contacts.sum().item() / 2 / n_pairs
    p_dca = co_eq.precision(co_eq.top_pairs(co_solve.mfdca_scores(x, 0.3), L), contacts)
    p_cov = co_eq.precision(co_eq.top_pairs(co_solve.covariance_scores(x, 0.3), L), contacts)
    assert random_level < 0.1
    assert p_dca > 0.75
    assert p_dca >= p_cov  # removing indirect correlations helps


def test_pseudo_likelihood_also_finds_the_contacts(planted_msa):
    x, contacts = planted_msa
    L = contacts.shape[0]
    scores, _, _, losses = co_solve.train_plm(x, 150, 0.05, 1e-4, n_snapshots=3)
    assert losses[-1] < losses[0]
    assert co_eq.precision(co_eq.top_pairs(scores, L), contacts) > 0.7


# --- single-cell PCA ---


@pytest.fixture(scope="module")
def cells():
    mu, markers = sc_t.type_profiles(800, 4, 40, 4.0, 0, CPU)
    counts, labels, _ = sc_t.sample_counts(600, mu, 0.3, 0.5, 1)
    return counts, labels, markers


def test_negative_binomial_counts_are_overdispersed():
    mu = torch.full((1, 50), 5.0)
    counts, _, _ = sc_t.sample_counts(4000, mu, 0.5, 0.0, 3)
    m, v = counts.mean(), counts.var(0).mean()
    assert m.item() == pytest.approx(5.0, rel=0.05)
    assert v.item() == pytest.approx(5.0 + 0.5 * 25.0, rel=0.1)  # m + φ m²


def test_normalisation_removes_library_size(cells):
    counts, _, _ = cells
    scaled, X = sc_eq.normalise(counts)
    assert torch.allclose(scaled.sum(1), torch.full((len(counts),), 1e4))
    assert torch.allclose(X, torch.log1p(scaled))


def test_pca_explained_variance_is_sorted_and_sums_to_at_most_one(cells):
    counts, _, _ = cells
    scaled, X = sc_eq.normalise(counts)
    hvg = torch.argsort(sc_eq.hvg_scores(scaled), descending=True)[:150]
    y, V, ratio = sc_eq.pca(X[:, hvg], 10)
    assert (ratio[1:] <= ratio[:-1] + 1e-7).all()
    assert ratio[:10].sum() <= 1.0
    assert ratio.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(V.T @ V, torch.eye(10), atol=1e-4)  # orthonormal loadings
    # the score variance along each PC is its share of the total variance
    Z = X[:, hvg] - X[:, hvg].mean(0)
    assert torch.allclose(y.var(0) / Z.var(0).sum(), ratio[:10], rtol=1e-3)


def test_hvgs_are_enriched_in_marker_genes(cells):
    counts, _, markers = cells
    scaled, _ = sc_eq.normalise(counts)
    hvg = torch.argsort(sc_eq.hvg_scores(scaled), descending=True)[:100]
    hit = torch.isin(hvg, markers.flatten()).float().mean().item()
    assert hit > 3 * markers.numel() / counts.shape[1]  # far above the 20% base rate


def test_kmeans_never_increases_the_inertia(cells):
    counts, _, _ = cells
    _, X = sc_eq.normalise(counts)
    y, _, _ = sc_eq.pca(X, 8)
    km = sc_solve.kmeans(y, 5, 6, 15, 0)
    inertia = km["inertia"]  # (T+1, R)
    assert (inertia[1:] <= inertia[:-1] * (1 + 1e-6) + 1e-6).all()
    assert km["labels"].shape == (16, y.shape[0])


def test_ari_is_one_for_relabelled_partitions_and_near_zero_for_random():
    gen = torch.Generator().manual_seed(4)
    a = torch.randint(4, (500,), generator=gen)
    perm = torch.tensor([2, 0, 3, 1])
    assert sc_eq.adjusted_rand_index(a, perm[a]) == pytest.approx(1.0)
    b = torch.randint(4, (500,), generator=gen)
    assert abs(sc_eq.adjusted_rand_index(a, b)) < 0.05


def test_ari_by_brute_force_pair_counting():
    a = torch.tensor([0, 0, 1, 1, 2, 2, 2, 0])
    b = torch.tensor([1, 1, 0, 0, 0, 2, 2, 2])
    same_a = [a[i] == a[j] for i, j in itertools.combinations(range(8), 2)]
    same_b = [b[i] == b[j] for i, j in itertools.combinations(range(8), 2)]
    n11 = sum(bool(x and y) for x, y in zip(same_a, same_b))
    na, nb, n = sum(map(bool, same_a)), sum(map(bool, same_b)), len(same_a)
    expected = (n11 - na * nb / n) / (0.5 * (na + nb) - na * nb / n)
    assert sc_eq.adjusted_rand_index(a, b) == pytest.approx(expected)


def test_pipeline_ari_is_high_on_well_separated_types(cells):
    counts, labels, _ = cells
    scaled, X = sc_eq.normalise(counts)
    hvg = torch.argsort(sc_eq.hvg_scores(scaled), descending=True)[:200]
    y, _, _ = sc_eq.pca(X[:, hvg], 10)
    km = sc_solve.kmeans(y, 4, 8, 15, 0)
    assert sc_eq.adjusted_rand_index(km["labels"][-1], labels) > 0.9


# --- DNA-binding CNN ---


def test_convolution_matches_torch_conv1d():
    gen = torch.Generator().manual_seed(5)
    x = torch.nn.functional.one_hot(torch.randint(4, (3, 30), generator=gen), 4).float()
    W, c = torch.randn(5, 7, 4, generator=gen), torch.randn(5, generator=gen)
    ref = torch.nn.functional.conv1d(x.transpose(1, 2), W.transpose(1, 2), c)  # (N, F, P)
    assert torch.allclose(dna_eq.conv(x, W, c), ref, atol=1e-5)


def test_motif_similarity_and_information():
    pwm = dna_t.motif_pwm("GGGACTTTCC", 0.9, CPU)
    r, shift = dna_eq.motif_similarity(pwm, pwm)
    assert r == pytest.approx(1.0) and shift == 0
    padded = torch.cat([torch.full((2, 4), 0.25), pwm, torch.full((3, 4), 0.25)])
    assert dna_eq.motif_similarity(padded, pwm) == (pytest.approx(1.0), 2)
    assert torch.allclose(dna_eq.information(torch.full((3, 4), 0.25)), torch.zeros(3), atol=1e-6)
    assert dna_eq.information(torch.eye(4)).allclose(torch.full((4,), 2.0))


def test_planted_sites_are_in_the_positive_sequences():
    pwm = dna_t.motif_pwm("TGACTCA", 1.0, CPU)
    x, y, starts = dna_t.make_dataset(200, 40, pwm, 0, CPU)
    seqs = x.argmax(-1)
    word = torch.tensor(["ACGT".index(b) for b in "TGACTCA"])
    for n in torch.nonzero(y).flatten().tolist():
        assert torch.equal(seqs[n, starts[n] : starts[n] + 7], word)
    assert 0.4 < y.mean().item() < 0.6


@pytest.fixture(scope="module")
def trained_cnn():
    motif = dna_t.motif_pwm("GGGACTTTCC", 0.95, CPU)
    x, y, _ = dna_t.make_dataset(2500, 50, motif, 0, CPU)
    x_tr, y_tr, x_te, y_te = x[500:], y[500:], x[:500], y[:500]
    weights = dna_t.init_weights(4, 12, 0, CPU)
    hist = dna_solve.train(weights, x_tr, y_tr, x_te, y_te, 8, 0.01, 100, 0, n_snapshots=5)
    return motif, weights, hist, x_te, y_te


def test_cnn_reaches_high_accuracy(trained_cnn):
    _, weights, hist, x_te, y_te = trained_cnn
    assert hist["test_loss"][-1] < 0.5 * hist["test_loss"][0]
    assert dna_eq.accuracy(dna_eq.forward(x_te, weights), y_te) > 0.9


def test_best_filter_recovers_the_planted_motif(trained_cnn):
    motif, weights, _, x_te, y_te = trained_cnn
    with torch.no_grad():
        m = torch.relu(dna_eq.conv(x_te, weights["W"], weights["c"])).amax(2)
        importance = weights["v"] * (m[y_te == 1].mean(0) - m[y_te == 0].mean(0))
        pwms = dna_eq.filter_pwms(x_te[y_te == 1], weights["W"], weights["c"])
    r, _ = dna_eq.motif_similarity(pwms[int(importance.argmax())], motif)
    assert r > 0.9
