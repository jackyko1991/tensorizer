"""Main loop: planted Potts family → Gibbs sampling → masked-LM training → contacts from attention → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

ROWS, COLS = 5, 6  # lattice fold: L = 30 residues
N_TEST = 500


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    q = params["q"]
    coords = tensors.lattice_fold(ROWS, COLS, dev)  # (L, 2)
    contacts = tensors.contact_map(coords)  # (L, L) bool
    L, n_contacts = coords.shape[0], int(torch.triu(contacts, 1).sum())
    h, J = tensors.planted_potts(contacts, q, params["coupling"], 0.5, seed=0)

    n_seq = 800 if fast else params["n_seq"]
    seqs = solve.gibbs_sample(h, J, tensors.parity(coords), n_seq + N_TEST, 20 if fast else 120, seed=0)  # (N, L)
    train, test = seqs[:n_seq], seqs[n_seq:]
    ce_uni, ce_prof = solve.baselines(train, test, q)
    rand = solve.random_precision(contacts)
    print(
        f"family: {n_seq} training + {N_TEST} test sequences, L = {L}, q = {q} letters, {n_contacts} planted contacts"
    )
    print(f"baselines on test letters: unigram {ce_uni:.3f} nats, per-site profile {ce_prof:.3f} nats")

    torch.manual_seed(0)
    layers, heads, d = params["layers"], params["heads"], params["d_model"]
    w = tensors.init_weights(L, q, d, heads, layers, seed=0, device=dev)
    n_params = sum(t.numel() for t in tensors.parameters(w))
    n_steps = 60 if fast else params["steps"]
    print(f"transformer: {layers} layer(s) × {heads} heads, d = {d}, {n_params} weights")
    hist = solve.train(train, test, contacts, w, q, n_steps, 128, params["lr"], params["mask_rate"],
                       n_snapshots=6 if fast else 40)  # fmt: skip
    for k in range(0, len(hist["at"]), max(1, len(hist["at"]) // 6)):
        print(
            f"  step {hist['at'][k]:5d}  test loss {hist['test'][k]:.3f}  contact precision {hist['precision'][k]:.2f}"
        )
    print(f"final: test loss {hist['test'][-1]:.3f} nats (profile {ce_prof:.3f}), ", end="")
    print(f"precision@{n_contacts} = {hist['precision'][-1]:.2f} (random {rand:.3f})")

    with torch.no_grad():
        _, maps = equation.forward(test[:8], w)
    print(f"attention rows sum to 1: max |Σ_j A_ij − 1| = {(maps.sum(-1) - 1).abs().max():.1e}")
    return plot.animate(hist, contacts, n_contacts, ce_uni, ce_prof, rand, out_dir / "protein_language_model.gif")


if __name__ == "__main__":
    cli(run, __file__)
