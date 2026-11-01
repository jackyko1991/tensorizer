"""Main loop: simulate charged particles → train an EGNN and a plain MLP → check equivariance → roll out → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors

K_SPRING = 0.3  # weak springs between every pair keep the cloud together
EPS = 0.3  # softening of the Coulomb force at short range
N_STEPS = 25  # time steps per training trajectory
N_TEST = 8  # held-out trajectories
HIDDEN = 32  # EGNN feature width
N_LAYERS = 3
MLP_WIDTH = 128
SEED = 0


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator().manual_seed(SEED)
    N, dt = params["n_particles"], params["dt"]
    n_iter = 60 if fast_mode() else params["n_iter"]

    # 1. Data: RK4 trajectories of random systems, cut into (state, next state) pairs.
    x0, v0, c = tensors.initial_states(params["n_train"] + N_TEST, N, gen, dev)
    traj = solve.simulate(x0, v0, c, K_SPRING, EPS, dt, N_STEPS)  # (T + 1, B, 2, N, 3)
    n_tr = params["n_train"]
    train_set = solve.pairs(traj[:, :n_tr], c[:n_tr])
    test_set = solve.pairs(traj[:, n_tr:], c[n_tr:])
    now, nxt, _ = test_set
    drift = ((now[:, 0] + dt * now[:, 1] - nxt[:, 0]) ** 2).mean() + ((now[:, 1] - nxt[:, 1]) ** 2).mean()
    print(f"{n_tr} training systems × {N_STEPS} steps = {len(train_set[0])} pairs, {N} particles in 3D, dt = {dt}")
    print(f"reference: 'no forces' guess (x + dt·v, v) has test MSE {drift.item():.2e}")

    # 2. Train both models on the same pairs.
    net_egnn = tensors.egnn_weights(HIDDEN, N_LAYERS, gen, dev)
    net_mlp = tensors.mlp_weights([7 * N, MLP_WIDTH, MLP_WIDTH, 6 * N], gen, dev)
    results = {}
    for name, net in (("egnn", net_egnn), ("mlp", net_mlp)):
        n_params = sum(p.numel() for p in tensors.parameters(net))
        hist = solve.train(name, net, tensors.parameters(net), train_set, test_set, dt, n_iter, params["lr"])
        results[name] = hist
        print(f"{name.upper():4s} {n_params:6d} weights   train MSE {hist['train_final']:.2e}   "
              f"test MSE {hist['test_final']:.2e}")  # fmt: skip

    # 3. Rotation check on held-out states: rotate + translate the input, compare with rotating the output.
    R, shift = tensors.random_rotation(gen, dev), torch.tensor([0.7, -0.3, 1.1], device=dev)
    for name, net in (("egnn", net_egnn), ("mlp", net_mlp)):
        err = solve.equivariance_error(name, net, now[:, 0], now[:, 1], test_set[2], R, shift, dt)
        print(f"{name.upper():4s} equivariance error |f(Rx+s) - (R f(x)+s)| / |f| = {err:.1e}")

    # 4. Roll out one held-out system for longer than the training trajectories.
    n_roll = params["rollout_steps"]
    b = n_tr  # first test system
    true = solve.simulate(x0[b : b + 1], v0[b : b + 1], c[b : b + 1], K_SPRING, EPS, dt, n_roll)[:, 0, 0]  # (T, N, 3)
    pred = {
        name: solve.rollout(name, net, x0[b : b + 1], v0[b : b + 1], c[b : b + 1], dt, n_roll)[:, 0]
        for name, net in (("egnn", net_egnn), ("mlp", net_mlp))
    }
    for name, xs in pred.items():
        err = (xs - true).norm(dim=-1).mean(dim=-1)  # mean particle position error at each step
        print(f"{name.upper():4s} rollout position error after {n_roll} steps: {err[-1].item():.3f}")

    return plot.animate(true, pred, c[b], results, dt, out_dir / "egnn_dynamics.gif")


if __name__ == "__main__":
    cli(run, __file__)
