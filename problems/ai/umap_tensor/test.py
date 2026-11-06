"""Checks written first (TDD): what any correct UMAP-as-tensors code must satisfy.

Run with `pytest problems/ai/umap_tensor/test.py`. Each test is a mathematical property, not an implementation detail:
the k-nearest neighbours equal a brute-force search, every σ_i meets its target sum, the fuzzy union is a symmetric
membership in [0, 1], the fitted curve reproduces the target, a step lowers the cross-entropy, the hand-written forces
are the gradient of the sampled loss, the metrics equal their definitions, and the final layouts keep neighbourhoods.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_knn_equals_brute_force():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(60, 5, generator=g, dtype=torch.float64)
    dist, idx = equation.knn(x, 7)  # (60, 7), self first
    for i in range(60):  # slow reference: every distance from point i, sorted
        d = [math.dist(x[i].tolist(), x[j].tolist()) for j in range(60)]
        order = sorted(range(60), key=lambda j: d[j])[:7]
        assert idx[i].tolist() == order
        assert dist[i].tolist() == pytest.approx([d[j] for j in order], abs=1e-9)
    db, ib = equation.knn(x, 7, block=16)  # blockwise (memory-light) version gives the same graph
    assert torch.equal(ib, idx) and torch.allclose(db, dist)


@pytest.mark.parametrize("k", [5, 15, 30])
def test_sigma_meets_target_sum_for_every_point(k):
    x, _ = tensors.mnist(CPU, n=600)
    dist, _ = equation.knn(x, k)
    rho, sigma = equation.smooth_knn(dist, k)  # (N,), (N,)
    assert torch.allclose(rho, dist[:, 1])  # ρ_i = distance to the nearest other point (no duplicates here)
    s = torch.exp(-(dist[:, 1:] - rho[:, None]).clamp(min=0) / sigma[:, None]).sum(1)  # (N,)
    assert (s - math.log2(k)).abs().max() < 1e-4 * math.log2(k)  # Σ_j exp(-(d_ij - ρ_i)/σ_i) = log2 k, all i


def test_fuzzy_union_is_symmetric_membership():
    x, _ = tensors.swiss_roll(CPU, n=400)[:2]
    g = solve.fuzzy_graph(x, 15)
    P, W = g["P"], g["W"]  # (N, N) directed memberships and their fuzzy union
    assert torch.equal(W, W.T)
    assert W.min() >= 0 and W.max() <= 1 + 1e-6
    assert (W >= torch.maximum(P, P.T) - 1e-6).all()  # a probabilistic OR is at least each of its inputs
    assert torch.allclose(W, P + P.T - P * P.T)
    assert torch.allclose(P.max(1).values, torch.ones(len(x)))  # every point's nearest neighbour has membership 1
    assert (W.diagonal() == 0).all()


def test_ab_fit_reproduces_target_curve():
    a, b = equation.fit_ab(min_dist=0.1, spread=1.0)
    x = torch.linspace(0, 3, 300, dtype=torch.float64)
    target = torch.where(x < 0.1, torch.ones_like(x), torch.exp(-(x - 0.1)))
    assert (1 / (1 + a * x ** (2 * b)) - target).abs().max() < 0.06  # least-squares fit of a smooth curve to a kink
    assert a == pytest.approx(1.577, rel=0.01) and b == pytest.approx(0.895, rel=0.01)  # umap-learn's values
    ref = tensors.reference()
    if ref is not None:  # a, b printed by umap-learn's find_ab_params in the offline reference
        line = next(ln for ln in tensors.REF_CSV.read_text(encoding="utf-8").splitlines() if "find_ab_params" in ln)
        a_ref, b_ref = (float(p.split("=")[1].split(",")[0]) for p in line.split(":")[1].split(", "))
        assert a == pytest.approx(a_ref, rel=2e-3) and b == pytest.approx(b_ref, rel=2e-3)


def test_one_step_decreases_cross_entropy():
    x, _ = tensors.mnist(CPU, n=100)
    g = solve.fuzzy_graph(x.double(), 10)
    a, b = equation.fit_ab(0.1, 1.0)
    y = torch.randn(len(x), 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 3
    ce0 = equation.cross_entropy(g["W"], y, a, b)
    y1 = solve.expected_step(y, g["W"], a, b, lr=0.01)  # all edges, repulsion from every non-edge pair
    assert equation.cross_entropy(g["W"], y1, a, b) < ce0


def test_hand_forces_are_the_gradient_of_the_sampled_loss():
    g = torch.Generator().manual_seed(3)
    y = torch.randn(20, 2, generator=g, dtype=torch.float64, requires_grad=True)
    i, j, n = torch.arange(0, 10), torch.arange(10, 20), torch.randint(0, 20, (10,), generator=g)
    a, b = 1.577, 0.895
    loss = equation.sampled_loss(y, i, j, i, n, a, b)  # Σ -log q_ij + Σ -log(1 - q_in), q_in w.r.t. y_i only
    (grad,) = torch.autograd.grad(loss, y)
    force = equation.forces(y.detach(), i, j, i, n, a, b, clip=None, eps=0.0)  # hand-written: -∂loss/∂y
    assert torch.allclose(-grad, force, atol=1e-6)


def test_metrics_equal_their_definitions():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(40, 6, generator=g, dtype=torch.float64)
    y = x[:, :2] + 0.5 * torch.randn(40, 2, generator=g, dtype=torch.float64)
    k, n = 5, 40

    def ranks(z):  # rank of j among i's neighbours (1 = nearest), plain loops
        out = [[0] * n for _ in range(n)]
        for i in range(n):
            order = sorted((j for j in range(n) if j != i), key=lambda j: math.dist(z[i].tolist(), z[j].tolist()))
            for r, j in enumerate(order, 1):
                out[i][j] = r
        return out

    rx, ry = ranks(x), ranks(y)
    pen_t = sum(rx[i][j] - k for i in range(n) for j in range(n) if 0 < ry[i][j] <= k and rx[i][j] > k)
    pen_c = sum(ry[i][j] - k for i in range(n) for j in range(n) if 0 < rx[i][j] <= k and ry[i][j] > k)
    norm = 2 / (n * k * (2 * n - 3 * k - 1))
    assert equation.trustworthiness(x, y, k) == pytest.approx(1 - norm * pen_t)
    assert equation.continuity(x, y, k) == pytest.approx(1 - norm * pen_c)
    shared = sum(len({j for j in range(n) if 0 < rx[i][j] <= k} & {j for j in range(n) if 0 < ry[i][j] <= k})
                 for i in range(n))  # fmt: skip
    assert equation.knn_recall(x, y, k) == pytest.approx(shared / (n * k))
    assert equation.trustworthiness(x, x, k) == pytest.approx(1.0)


def test_metrics_match_scikit_learn_on_the_reference_embedding():
    ref = tensors.reference()
    if ref is None:
        pytest.skip("no bundled umap-learn reference")
    x, _, _ = tensors.swiss_roll(CPU)
    assert float(x.double().sum()) == pytest.approx(float(ref["swiss_checksum"]), rel=1e-6)  # same points
    xm, _ = tensors.mnist(CPU)
    for name, xx in (("swiss_roll", x), ("mnist", xm)):
        row = next(
            r for r in ref["rows"] if r["dataset"] == name and r["method"] == "umap-learn" and r["run"] == "final"
        )
        t = equation.trustworthiness(xx, torch.from_numpy(ref[name]), 10)
        assert t == pytest.approx(float(row["sk_trustworthiness"]), abs=2e-4)  # sklearn.manifold.trustworthiness


def test_spectral_init_separates_two_clusters():
    g = torch.Generator().manual_seed(5)
    x = torch.cat([torch.randn(50, 3, generator=g), torch.randn(50, 3, generator=g) + 8.0])
    graph = solve.fuzzy_graph(x, 10)
    y = solve.spectral_init(graph["W"], 2, seed=0)  # (100, 2), Laplacian eigenvectors
    lo = torch.minimum(y[:50].max(0).values - y[50:].min(0).values, y[50:].max(0).values - y[:50].min(0).values)
    assert (lo < 0).any()  # along one eigenvector the two components do not overlap at all


def test_swiss_roll_layout_keeps_neighbourhoods():
    x, u, _ = tensors.swiss_roll(CPU, n=800)
    out = solve.umap(x, n_epochs=200, seed=0)
    assert equation.trustworthiness(x, out["Y"], 10) > 0.95
    assert equation.trustworthiness(x, solve.pca(x), 10) < equation.trustworthiness(x, out["Y"], 10)


@pytest.mark.slow
def test_mnist_layout_is_trustworthy_and_separates_digits():
    x, y = tensors.mnist(CPU)
    for mode in ("forces", "autograd"):
        out = solve.umap(x, n_epochs=200, seed=0, mode=mode)
        assert equation.trustworthiness(x, out["Y"], 10) > 0.93  # umap-learn: about 0.96 (headroom for BLAS builds)
        assert equation.knn_accuracy(out["Y"], y, 10) > 0.85  # PCA in 2D: about 0.45
    p = solve.pca(x)
    assert equation.knn_accuracy(p, y, 10) < 0.6


@pytest.mark.slow
def test_full_run(tmp_path):
    from problems._common.runtime import load_meta, resolve_params

    from . import main

    out = main.run(resolve_params(load_meta(main.__file__.rsplit("main.py", 1)[0])), tmp_path)
    assert out.exists() and (tmp_path / "results" / "metrics.csv").is_file()
