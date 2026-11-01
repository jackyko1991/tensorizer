"""Main loop: random molecules → Hückel targets → train GNN and MLP → test on larger molecules → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

N_PAD = 16  # every molecule is padded to 16 atoms (the largest test size)
TRAIN_SIZES = (4, 10)  # atoms per training molecule
LARGE_SIZES = (11, 16)  # extrapolation test: larger molecules than any seen in training
BATCH = 128
SEED = 0


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    n_train = 400 if fast else params["n_train"]
    n_test = 100 if fast else 400
    epochs = 10 if fast else params["epochs"]
    t0 = time.perf_counter()

    # --- data: graphs and their exact Hückel π energies ---
    train = tensors.dataset(n_train, TRAIN_SIZES, N_PAD, SEED, dev)
    small = tensors.dataset(n_test, TRAIN_SIZES, N_PAD, SEED + 1, dev)  # same sizes, unseen molecules
    large = tensors.dataset(n_test, LARGE_SIZES, N_PAD, SEED + 2, dev)  # bigger molecules
    y = {
        name: equation.huckel_energy(d["A"], d["mask"])
        for name, d in [("train", train), ("small", small), ("large", large)]
    }
    tests = {"small": small, "large": large}

    # --- train the GNN and the MLP on the same molecules ---
    w_gnn = tensors.init_gnn(params["width"], params["layers"], SEED, dev)
    w_mlp = tensors.init_mlp(N_PAD, 128, SEED, dev)
    t1 = time.perf_counter()
    h_gnn = solve.train(
        equation.gnn, w_gnn, {**train, "y": y["train"]}, tests, epochs, params["lr"], BATCH, SEED, "GNN epochs"
    )
    t_gnn = time.perf_counter() - t1
    t1 = time.perf_counter()
    h_mlp = solve.train(equation.mlp, w_mlp, {**train, "y": y["train"]}, tests, epochs, 1e-3, BATCH, SEED, "MLP epochs")
    t_mlp = time.perf_counter() - t1

    # --- baselines and the permutation test ---
    _, fit = equation.count_baseline(train, y["train"])
    perm = torch.randperm(N_PAD, generator=torch.Generator().manual_seed(SEED + 3)).to(dev)
    with torch.no_grad():
        y_gnn, e_gnn = equation.gnn(large, w_gnn, return_atoms=True)
        gnn_perm = (equation.gnn(tensors.permute(large, perm), w_gnn) - y_gnn).abs().max().item()
        mlp_perm = (equation.mlp(tensors.permute(large, perm), w_mlp) - equation.mlp(large, w_mlp)).abs().max().item()

    def mae(pred, name):
        return (pred - y[name]).abs().mean().item()

    print(f"{n_train} training molecules with {TRAIN_SIZES[0]}–{TRAIN_SIZES[1]} carbons, test sets of {n_test}: "
          f"same sizes and {LARGE_SIZES[0]}–{LARGE_SIZES[1]} carbons   ·   y = π energy in units of |β|")  # fmt: skip
    print(f"GNN: {params['layers']} message-passing layers × {params['width']} channels, {tensors.count(w_gnn):,} "
          f"weights, {t_gnn:.1f} s   ·   MLP on the 16×16 adjacency: {tensors.count(w_mlp):,} weights, "
          f"{t_mlp:.1f} s")  # fmt: skip
    print("mean absolute error (|β|)        same-size test   larger molecules")
    rows = [
        ("predict the training mean", torch.full_like(y["small"], y["train"].mean().item()),
         torch.full_like(y["large"], y["train"].mean().item())),
        ("bond counting  a·N + b·N_bonds + c", equation.count_baseline(small, y["small"], fit)[0],
         equation.count_baseline(large, y["large"], fit)[0]),
        ("MLP on the padded adjacency", h_mlp["small"][-1].to(dev), h_mlp["large"][-1].to(dev)),
        ("message-passing GNN", h_gnn["small"][-1].to(dev), h_gnn["large"][-1].to(dev)),
    ]  # fmt: skip
    for name, p_small, p_large in rows:
        print(f"  {name:<34} {mae(p_small, 'small'):10.3f} {mae(p_large, 'large'):16.3f}")
    print(f"training MSE: GNN {h_gnn['loss'][0].item():.2f} → {h_gnn['loss'][-1].item():.4f},  MLP "
          f"{h_mlp['loss'][0].item():.2f} → {h_mlp['loss'][-1].item():.4f}   "
          f"(variance of y: {y['train'].var().item():.2f})")  # fmt: skip
    print(f"relabel the atoms of every large molecule: GNN output changes by ≤ {gnn_perm:.1e}, MLP by up to "
          f"{mlp_perm:.2f}   ·   total {time.perf_counter() - t0:.1f} s")  # fmt: skip

    m = int(torch.argmax(large["mask"].sum(-1) * 10 + large["A"].sum((-1, -2))))  # biggest, most-ringed molecule
    example = {"A": large["A"][m], "mask": large["mask"][m], "e": e_gnn[m], "y": y["large"][m].item(),
               "pred": y_gnn[m].item(), "gnn_perm": gnn_perm, "mlp_perm": mlp_perm}  # fmt: skip
    return plot.animate(h_gnn, h_mlp, y, y["train"].var().item(), example, out_dir / "molecular_gnn.gif")


if __name__ == "__main__":
    cli(run, __file__)
