"""Physics checks, written first (TDD): what any correct molecular graph-neural-network code must satisfy.

Run with `pytest problems/chemistry/molecular_gnn/test.py`. Each test is a physical law, not an implementation
detail: the Hückel target matches textbook values, molecules are valid sp² graphs, and energy ignores atom ordering.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _molecule(edges, n, n_pad=8):
    A = torch.zeros(1, n_pad, n_pad)
    for i, j in edges:
        A[0, i, j] = A[0, j, i] = 1.0
    mask = (A.sum(-1) > 0).float()
    return {"A": A, "mask": mask, "x": tensors.features(A, mask)}


def test_huckel_energy_matches_textbook_values():
    benzene = _molecule([(i, (i + 1) % 6) for i in range(6)], 6)
    butadiene = _molecule([(0, 1), (1, 2), (2, 3)], 4)
    allyl = _molecule([(0, 1), (1, 2)], 3)  # odd electron in the non-bonding orbital (λ = 0)
    assert equation.huckel_energy(benzene["A"], benzene["mask"]).item() == pytest.approx(8.0, abs=1e-5)  # 8|β|
    assert equation.huckel_energy(butadiene["A"], butadiene["mask"]).item() == pytest.approx(
        2 * (2 * math.cos(math.pi / 5) + 2 * math.cos(2 * math.pi / 5)), abs=1e-5
    )  # chain levels 2 cos(kπ/(N+1))
    assert equation.huckel_energy(allyl["A"], allyl["mask"]).item() == pytest.approx(2 * math.sqrt(2), abs=1e-5)


def test_random_skeletons_are_valid_carbon_graphs():
    data = tensors.dataset(200, (4, 12), 16, 0, CPU)
    A, mask = data["A"], data["mask"]
    assert torch.equal(A, A.transpose(1, 2))  # bonds are symmetric
    assert (A.sum(-1) <= 3).all()  # sp² carbon valence: at most three C neighbours
    assert ((A.sum(-1) > 0) == (mask > 0)).all()
    reach = torch.matrix_power(torch.eye(16) + A, 16) > 0  # connected: every real atom reaches every other
    real = mask[:, :, None] * mask[:, None, :] > 0
    assert reach[real].all()


def test_gnn_is_permutation_invariant_and_the_mlp_is_not():
    data = tensors.dataset(50, (5, 12), 16, 3, CPU)
    w_gnn, w_mlp = tensors.init_gnn(16, 3, 0, CPU), tensors.init_mlp(16, 32, 0, CPU)
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        moved = tensors.permute(data, perm)
        assert torch.allclose(equation.gnn(moved, w_gnn), equation.gnn(data, w_gnn), atol=1e-4)
        assert not torch.allclose(equation.mlp(moved, w_mlp), equation.mlp(data, w_mlp), atol=1e-2)
    assert torch.allclose(
        equation.huckel_energy(moved["A"], moved["mask"]), equation.huckel_energy(data["A"], data["mask"])
    )  # energy is invariant under relabelling atoms


def test_gnn_training_loss_falls_below_the_mean_and_bond_counting_baselines():
    train = tensors.dataset(600, (4, 10), 16, 0, CPU)
    y = equation.huckel_energy(train["A"], train["mask"])
    w = tensors.init_gnn(24, 3, 0, CPU)
    hist = solve.train(equation.gnn, w, {**train, "y": y}, {}, 40, 0.005, 64, 0, "test")
    loss = hist["loss"]
    count_pred, _ = equation.count_baseline(train, y)
    assert loss[-1] < 0.1 * loss[1]  # it trains
    assert loss[-1] < 0.02 * y.var()  # far better than predicting the mean
    assert loss[-1] < equation.mse(count_pred, y)  # and better than a linear fit in atoms and bonds
