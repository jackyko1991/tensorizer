"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import mean

N_FINE = 512  # ground-truth resolution
N_TRAIN, N_TEST = 32, 8  # training trajectories / held-out trajectories (different random seed)
T_TRAIN, T_TEST = 100, 150  # coarse steps: trained on t ≤ 2, tested out to t = 3


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    torch.manual_seed(0)  # the network's initial weights
    nu, n_coarse = params["nu"], int(params["n_coarse"])
    n_iter = 10 if fast_mode() else params["train_iters"]
    g = solve.setup(N_FINE, n_coarse, dev)
    print(f"fine grid {N_FINE} cells, coarse grid {n_coarse} cells ({g['factor']}× coarser), Δt = {solve.DT_COARSE}")

    # ground truth: fine solver, averaged onto the coarse cells
    _, truth = solve.ground_truth(
        tensors.random_initial(g["x_f"], N_TRAIN, seed=1), nu, g["dx_f"], g["factor"], T_TRAIN
    )
    fine_test, truth_test = solve.ground_truth(
        tensors.random_initial(g["x_f"], N_TEST, seed=2), nu, g["dx_f"], g["factor"], T_TEST
    )

    # learn the flux correction by differentiating through `unroll` coarse steps
    model = tensors.FluxCorrection().to(dev)
    n_weights = sum(p.numel() for p in model.parameters())
    losses = solve.train(model, truth, nu, g["dx_c"], n_iter, params["unroll"])
    print(f"CNN with {n_weights} weights, {n_iter} Adam steps: loss {losses[0]:.2e} → {losses[-1]:.2e}")

    # held-out initial conditions, run 50% longer than anything seen in training
    with torch.no_grad():
        plain = solve.rollout(truth_test[0], nu, g["dx_c"], T_TEST)  # (T, B, N_c)
        learned = solve.rollout(truth_test[0], nu, g["dx_c"], T_TEST, model)
    err_plain = solve.rms_error(plain, truth_test)  # (T, B)
    err_learned = solve.rms_error(learned, truth_test)
    for name, e in [("plain coarse", err_plain), ("learned coarse", err_learned)]:
        print(f"{name:>15}: RMS error vs fine (held-out mean): t = 1: {e[50].mean():.4f}, t = 3: {e[-1].mean():.4f}")
    better = (err_learned.mean(0) < err_plain.mean(0)).sum().item()
    print(f"learned beats plain on {better}/{N_TEST} unseen initial conditions")
    drift = (mean(learned) - mean(learned[0])).abs().max().item()
    print(f"mean ⟨u⟩ drift of the learned solver: {drift:.1e} (conserved: δF is a flux)")

    times = torch.arange(T_TEST + 1) * solve.DT_COARSE
    return plot.animate(
        g["x_f"], g["x_c"], times, fine_test, plain, learned, err_plain, err_learned, losses,
        out_dir / "ml_accelerated_cfd.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
