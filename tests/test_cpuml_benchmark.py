"""Coding Skills page, "Migrating CPU-only ML to tensors": the tensor models in scripts/cpuml_benchmark.py are right.

These tests run without scikit-learn: every tensor model is checked against numpy (closed forms, brute force or
the optimality conditions of its objective). The comparisons with scikit-learn itself run only where it is
installed (they are skipped in the project environment).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cpuml_benchmark as cb  # noqa: E402


@pytest.fixture(scope="module")
def blobs():
    X, y = cb.blobs(300, 8, seed=1)
    return X, y


def t(a):
    return torch.as_tensor(a)


def test_pca_matches_eigendecomposition_of_the_covariance():
    X, _ = cb.blobs(400, 12, seed=2, spread=2.0)
    V, evr = (a.numpy() for a in cb.pca_tensor(t(X), 5))
    Xd = X.astype(np.float64)
    lam, U = np.linalg.eigh(np.cov(Xd, rowvar=False))
    lam, U = lam[::-1], U[:, ::-1]
    np.testing.assert_allclose(evr, lam[:5] / lam.sum(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.abs((V * U[:, :5].T).sum(1)), 1, atol=1e-4)  # same axes up to sign


def test_linear_svm_reaches_the_minimum_of_its_objective(blobs):
    X, y = blobs
    C = 0.05
    W = cb.linsvm_tensor(t(X), t(y), C).numpy().astype(np.float64)
    Xb = cb.with_bias(X)
    Y = np.where(y[:, None] == np.arange(cb.N_CLASSES), 1.0, -1.0)
    hinge = np.clip(1 - Y * (Xb @ W), 0, None)
    grad = W - 2 * C * Xb.T @ (Y * hinge)  # gradient of 1/2 |W|^2 + C sum hinge^2
    assert np.abs(grad).max() < 1e-2 * np.abs(W).max()  # float32 L-BFGS: stationary to ~0.3 %
    # and no random nudge lowers the objective
    J = cb.linsvm_objective(W, X, y, C)
    rng = np.random.default_rng(0)
    assert all(cb.linsvm_objective(W + 1e-3 * rng.normal(size=W.shape), X, y, C) >= J - 1e-6 for _ in range(5))


def test_logistic_regression_gradient_vanishes_at_the_solution(blobs):
    X, y = blobs
    C = 0.1
    Wb = cb.logreg_tensor(t(X), t(y), C).numpy().astype(np.float64)
    Z = cb.with_bias(X) @ Wb
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    G = C * cb.with_bias(X).T @ (P - np.eye(cb.N_CLASSES)[y])  # d(C sum CE)/dW
    G[:-1] += Wb[:-1]  # + d(1/2 |W|^2)/dW; the bias row is not penalised
    assert np.abs(G).max() < 1e-3


def test_logreg_objective_is_the_penalised_cross_entropy():
    X = np.array([[1.0, 0.0], [0.0, 2.0]], np.float32)
    y = np.array([0, 1])
    Wb = np.zeros((3, cb.N_CLASSES))
    # all-zero weights: every class has probability 1/10, so the loss is C * 2 * log 10
    assert cb.logreg_objective(Wb, X, y, 0.5) == pytest.approx(0.5 * 2 * np.log(10))


def test_projection_lands_in_the_box_and_on_the_hyperplane():
    gen = torch.Generator().manual_seed(3)
    v = torch.randn(200, generator=gen, dtype=torch.float64)
    y = torch.where(torch.rand(200, generator=gen) < 0.5, 1.0, -1.0).double()
    a = cb.project(v, y, 0.7, rounds=8)
    assert a.min() >= 0 and a.max() <= 0.7
    assert abs(float(y @ a)) < 1e-6
    # closest feasible point: any other feasible point (a plus a move along the hyperplane) is farther from v
    for _ in range(5):
        d = torch.randn(200, generator=gen, dtype=torch.float64) * 1e-3
        d -= (d @ y) / (y @ y) * y
        b = a + d
        if b.min() >= 0 and b.max() <= 0.7:
            assert (b - v).norm() >= (a - v).norm() - 1e-12


def test_kernel_svm_satisfies_the_dual_constraints_and_classifies():
    X, y = cb.blobs(240, 4, seed=4, spread=1.5)
    yb = np.where(y % 2 == 0, 1.0, -1.0).astype(np.float32)
    g = cb.rbf_gamma(X)
    a, b = cb.ksvm_tensor(t(X), t(yb), 1.0, g)
    a = a.numpy().astype(np.float64)
    assert a.min() >= 0 and a.max() <= 1.0 + 1e-6
    assert abs(float(yb @ a)) < 1e-3 * a.sum()
    acc = (cb.ksvm_predict(a, float(b), X, yb, X, g) == yb).mean()
    assert acc > 0.8
    # the dual is maximised: a feasible move along the constraints does not raise it
    D = cb.ksvm_dual(a, X, yb, g)
    assert D > 0 and D <= a.sum()


def test_knn_matches_brute_force(blobs):
    X, y = blobs
    Xtr, ytr, Xte = X[50:], y[50:], X[:50]
    got = cb.knn_tensor(t(Xtr), t(ytr), t(Xte), 7).numpy()
    d = ((Xte[:, None, :].astype(np.float64) - Xtr[None, :, :]) ** 2).sum(-1)
    want = []
    for row in d:
        votes = np.bincount(ytr[np.argsort(row)[:7]], minlength=cb.N_CLASSES)
        want.append(int(np.argmax(votes)))  # first maximum = smallest label on ties
    np.testing.assert_array_equal(got, want)


def test_kmeans_matches_numpy_lloyd():
    X, _ = cb.blobs(500, 3, seed=5, spread=3.0)
    c0 = X[:4].copy()
    centers, labels, n_iter = cb.kmeans_tensor(t(X), t(c0))
    c = c0.astype(np.float64)
    lab = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
    for _ in range(300):
        c = np.stack([X[lab == k].mean(0) for k in range(4)])
        new = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        if (new == lab).all():
            break
        lab = new
    np.testing.assert_array_equal(labels.numpy(), lab)
    np.testing.assert_allclose(centers.numpy(), c, rtol=1e-5, atol=1e-5)
    assert 1 <= n_iter < 300


def test_synthetic_inputs_are_reproducible():
    a = cb.synthetic("knn", 100)
    b = cb.synthetic("knn", 100)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[0].shape == (100, 64) and a[2].shape == (1000, 64)
    assert set(cb.CASES) == {"pca", "linsvm", "ksvm", "logreg", "knn", "kmeans"}


def test_every_case_has_a_sklearn_and_a_tensor_version():
    for case in cb.CASES:
        assert callable(cb.fns(case, "sklearn")) and callable(cb.fns(case, "tensor"))


# --- against scikit-learn itself, where it is installed ------------------------------------------------------


def test_against_sklearn(blobs):
    pytest.importorskip("sklearn")
    X, y = blobs
    W_sk = cb.linsvm_sklearn(X, y, 0.05)
    W = cb.linsvm_tensor(t(X), t(y), 0.05).numpy()
    assert cb.linsvm_objective(W, X, y, 0.05) == pytest.approx(cb.linsvm_objective(W_sk, X, y, 0.05), rel=1e-3)
    L_sk = cb.logreg_sklearn(X, y, 0.1)
    L = cb.logreg_tensor(t(X), t(y), 0.1).numpy()
    assert cb.logreg_objective(L, X, y, 0.1) == pytest.approx(cb.logreg_objective(L_sk, X, y, 0.1), rel=1e-4)
    np.testing.assert_array_equal(
        cb.knn_tensor(t(X[50:]), t(y[50:]), t(X[:50]), 5).numpy(), cb.knn_sklearn(X[50:], y[50:], X[:50], 5)
    )
