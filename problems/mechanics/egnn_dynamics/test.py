"""Physics checks, written first (TDD): what any correct EGNN-dynamics code must satisfy.

Run with `pytest problems/mechanics/egnn_dynamics/test.py`. The true forces obey Newton's third law, the EGNN is
rotation and translation equivariant (an MLP is not), and a trained EGNN learns the forces.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _data(n_train: int, n_test: int = 6, n_steps: int = 20, dt: float = 0.1):
    gen = torch.Generator().manual_seed(1)
    x0, v0, c = tensors.initial_states(n_train + n_test, 5, gen, CPU)
    traj = solve.simulate(x0, v0, c, 0.3, 0.3, dt, n_steps)
    return gen, solve.pairs(traj[:, :n_train], c[:n_train]), solve.pairs(traj[:, n_train:], c[n_train:])


def test_true_forces_obey_newtons_third_law():
    gen = torch.Generator().manual_seed(0)
    x, _, c = tensors.initial_states(4, 6, gen, CPU)
    a = equation.acceleration(x.double(), c.double(), 0.3, 0.3)
    assert torch.allclose(a.sum(dim=1), torch.zeros(4, 3, dtype=torch.float64), atol=1e-12)  # total momentum fixed


def test_egnn_is_rotation_and_translation_equivariant_mlp_is_not():
    gen, _, (now, _, c) = _data(n_train=2)
    R = tensors.random_rotation(gen, CPU)
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6) and torch.det(R) > 0
    shift = torch.tensor([0.5, -1.0, 2.0])
    net = tensors.egnn_weights(16, 2, gen, CPU)
    mlp = tensors.mlp_weights([35, 64, 64, 30], gen, CPU)
    err_egnn = solve.equivariance_error("egnn", net, now[:, 0], now[:, 1], c, R, shift, 0.1)
    err_mlp = solve.equivariance_error("mlp", mlp, now[:, 0], now[:, 1], c, R, shift, 0.1)
    assert err_egnn < 1e-5
    assert err_mlp > 1e-2


@pytest.mark.slow  # long training/sampling run: CI only, skipped by the in-app TDD check
def test_trained_egnn_beats_mlp_on_held_out_systems():
    torch.manual_seed(0)
    gen, train_set, test_set = _data(n_train=8)
    net = tensors.egnn_weights(32, 3, gen, CPU)
    mlp = tensors.mlp_weights([35, 128, 128, 30], gen, CPU)
    h_egnn = solve.train("egnn", net, tensors.parameters(net), train_set, test_set, 0.1, 250, 3e-3)
    h_mlp = solve.train("mlp", mlp, tensors.parameters(mlp), train_set, test_set, 0.1, 250, 3e-3)
    now, nxt, _ = test_set
    drift = ((now[:, 0] + 0.1 * now[:, 1] - nxt[:, 0]) ** 2).mean() + ((now[:, 1] - nxt[:, 1]) ** 2).mean()
    assert h_egnn["test_final"] < h_mlp["test_final"]
    assert h_egnn["test_final"] < 0.5 * drift.item()  # it learned the forces, not just x + v dt
