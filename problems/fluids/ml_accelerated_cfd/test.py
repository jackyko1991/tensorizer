"""Physics checks, written first (TDD): what any correct learned-flux Burgers code must satisfy.

Run with `pytest problems/fluids/ml_accelerated_cfd/test.py`. The finite-volume form conserves the mean of u even
with an arbitrary learned flux correction, coarse-graining keeps the mean, and the trained correction beats the
plain coarse solver on an unseen initial condition.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_finite_volume_conserves_mean():
    x, dx = tensors.grid(64, CPU)
    u0 = tensors.random_initial(x, 4, seed=3).double()
    plain = solve.rollout(u0, 0.02, dx, 200)
    drift = (equation.mean(plain) - equation.mean(u0)).abs().max()
    assert drift < 1e-12  # conservation law: flux differences telescope on a periodic grid
    model = tensors.FluxCorrection().double()
    torch.nn.init.normal_(model.net[-1].weight, std=0.1)  # any correction, even an untrained random one
    with torch.no_grad():
        learned = solve.rollout(u0, 0.02, dx, 50, model)
    assert (equation.mean(learned) - equation.mean(u0)).abs().max() < 1e-12  # still conservative by construction


def test_fine_solver_conserves_mean_and_untrained_model_is_plain():
    g = solve.setup(512, 32, CPU)
    u0 = tensors.random_initial(g["x_f"], 3, seed=4)
    fine, coarse = solve.ground_truth(u0, 0.02, g["dx_f"], g["factor"], 20)
    assert (equation.mean(fine) - equation.mean(u0)).abs().max() < 1e-5
    assert torch.allclose(equation.mean(coarse), equation.mean(fine), atol=1e-6)  # averaging keeps the mean
    model = tensors.FluxCorrection()
    with torch.no_grad():
        same = solve.rollout(coarse[0], 0.02, g["dx_c"], 10, model)
    assert torch.allclose(same, solve.rollout(coarse[0], 0.02, g["dx_c"], 10))  # last layer starts at zero


@pytest.mark.slow  # long training/sampling run: CI only, skipped by the in-app TDD check
def test_learned_solver_beats_plain_on_held_out_initial_condition():
    torch.manual_seed(0)
    nu = 0.02
    g = solve.setup(512, 32, CPU)
    _, truth = solve.ground_truth(tensors.random_initial(g["x_f"], 32, seed=1), nu, g["dx_f"], g["factor"], 100)
    _, test = solve.ground_truth(tensors.random_initial(g["x_f"], 4, seed=7), nu, g["dx_f"], g["factor"], 150)
    model = tensors.FluxCorrection()
    losses = solve.train(model, truth, nu, g["dx_c"], n_iter=150, unroll=12)
    assert losses[-1] < 0.3 * losses[0]
    with torch.no_grad():
        err_plain = solve.rms_error(solve.rollout(test[0], nu, g["dx_c"], 150), test).mean(0)
        err_learned = solve.rms_error(solve.rollout(test[0], nu, g["dx_c"], 150, model), test).mean(0)
    assert (err_learned < 0.6 * err_plain).all()  # every unseen run, out to t = 3 (training stopped at t = 2)
