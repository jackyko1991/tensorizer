"""Physics checks, written first (TDD): what any correct spectral clustering code must satisfy.

Run with `pytest problems/maths/spectral_clustering/test.py`. Each test is a mathematical invariant, not an
implementation detail: the Laplacian has zero row sums and is positive semidefinite, its zero eigenvalues count the
components, and a grid graph has the closed-form standing-wave spectrum.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def moons_graph(n: int = 300, noise: float = 0.08):
    x, truth = tensors.points("moons", n, noise, 0, CPU)
    return x, truth, equation.gaussian_knn_weights(equation.squared_distances(x), 0.2, 10)


def test_laplacian_rows_sum_to_zero_and_it_is_positive_semidefinite():
    _, _, w = moons_graph()
    assert torch.allclose(w, w.T)
    lap = equation.laplacian(w)
    assert lap.sum(dim=1).abs().max() < 1e-12
    assert torch.linalg.eigvalsh(lap).min() > -1e-10
    lsym = equation.normalized_laplacian(w)
    evals = torch.linalg.eigvalsh(lsym)
    assert evals.min() > -1e-10 and evals.max() < 2 + 1e-10
    f = torch.randn(len(w), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    assert torch.allclose(f @ lap @ f, equation.dirichlet_energy(w, f))


def test_zero_eigenvalue_multiplicity_equals_number_of_components():
    gen = torch.Generator().manual_seed(0)
    for sizes in ([5], [4, 6], [3, 3, 7], [2, 5, 4, 6]):
        n = sum(sizes)
        w = torch.zeros(n, n, dtype=torch.float64)
        start = 0
        for s in sizes:  # a random connected block: a path plus random extra edges
            block = torch.rand(s, s, generator=gen, dtype=torch.float64) * (torch.rand(s, s, generator=gen) < 0.4)
            block = block + torch.diag(torch.full((s - 1,), 0.5, dtype=torch.float64), 1)
            block = torch.triu(block, 1)
            w[start : start + s, start : start + s] = block + block.T
            start += s
        perm = torch.randperm(n, generator=gen)
        w = w[perm][:, perm]
        evals, _ = solve.spectrum(equation.laplacian(w))
        assert solve.count_zero_eigenvalues(evals) == len(sizes)
        assert solve.connected_components(w).unique().numel() == len(sizes)


def test_grid_graph_eigenvalues_match_the_standing_wave_formula():
    g = 12
    evals, _ = solve.spectrum(equation.laplacian(tensors.grid_graph(g, CPU)))
    one_d = 2 - 2 * torch.cos(math.pi * torch.arange(g, dtype=torch.float64) / g)
    exact = (one_d[:, None] + one_d[None, :]).flatten().sort().values
    assert torch.allclose(evals, exact, atol=1e-12)


def test_separates_moons_where_kmeans_fails():
    x, truth, w = moons_graph()
    spectral, _, _ = solve.spectral_clustering(w, 2, seed=0)
    plain = solve.kmeans(x, 2, seed=0)
    ari_spectral = solve.adjusted_rand_index(truth, spectral)
    ari_plain = solve.adjusted_rand_index(truth, plain)
    assert ari_spectral > 0.98
    assert ari_plain < 0.6
    lam2, fiedler = solve.fiedler_vector(w)
    assert lam2 > 0  # the graph is connected
    assert solve.adjusted_rand_index(truth, (fiedler > 0).long()) > 0.95


def test_adjusted_rand_index_limits():
    a = torch.tensor([0, 0, 1, 1, 2, 2])
    assert abs(solve.adjusted_rand_index(a, torch.tensor([2, 2, 0, 0, 1, 1])) - 1) < 1e-12
    random = torch.randint(0, 3, (3000,), generator=torch.Generator().manual_seed(0))
    other = torch.randint(0, 3, (3000,), generator=torch.Generator().manual_seed(1))
    assert abs(solve.adjusted_rand_index(random, other)) < 0.02
