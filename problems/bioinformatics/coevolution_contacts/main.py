"""Main loop: lattice fold → planted Potts model → Gibbs-sampled MSA → DCA (or pseudo-likelihood) → contacts → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import precision, top_pairs

FOLDS = {"3x3x4": (3, 3, 4), "4x4x4": (4, 4, 4), "4x4x5": (4, 4, 5)}
MIN_SEP = 3  # pairs closer than this along the chain are trivially close and not scored
SEED = 0


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    t0 = time.perf_counter()

    # --- the "protein": a compact lattice fold and its contacts ---
    coords = tensors.lattice_fold(FOLDS[params["fold"]], dev)  # (L, 3)
    contacts = tensors.contact_map(coords, MIN_SEP)  # (L, L) bool
    L, q = coords.shape[0], params["q"]
    n_contacts = int(contacts.sum()) // 2
    n_pairs = (L - MIN_SEP) * (L - MIN_SEP + 1) // 2
    print(f"lattice fold {params['fold']}: L = {L} residues, {n_contacts} contacts (|i−j| ≥ {MIN_SEP}), q = {q}")

    # --- evolution in a nutshell: sequences drawn from a Potts model with couplings on the contacts ---
    h, J = tensors.planted_potts(contacts, q, params["coupling"], 1.0, SEED)
    n_seq = 200 if fast else params["n_seq"]
    sweeps = 20 if fast else params["sweeps"]
    seqs = solve.gibbs_sample(h, J, tensors.parity(coords), n_seq, sweeps, SEED + 1)  # (N, L)
    x = tensors.one_hot(seqs, q)  # (N, L, q)
    print(f"MSA: {n_seq} sequences from {sweeps} checkerboard Gibbs sweeps ({time.perf_counter() - t0:.1f} s)")

    lam = params["pseudocount"]
    cov_full = solve.covariance_scores(x, lam)
    if params["method"] == "mfdca":
        # Frames: the same analysis on the first N sequences, N growing → how much data DCA needs.
        depths = torch.unique(torch.logspace(1.3, torch.log10(torch.tensor(n_seq)).item(), 6 if fast else 30).int())
        depths = depths.tolist()
        main_scores = torch.stack([solve.mfdca_scores(x[:n], lam).cpu() for n in depths])  # (S, L, L)
        cov_scores = torch.stack([solve.covariance_scores(x[:n], lam).cpu() for n in depths])
        xs, label = depths, "mean-field DCA"
    else:
        n_iter = 30 if fast else params["plm_iters"]
        _, xs, main_scores, losses = solve.train_plm(x, n_iter, 0.05, 1e-4)
        cov_scores = cov_full.cpu().expand(len(xs), L, L)
        label = "pseudo-likelihood"
        print(f"pseudo-likelihood: {n_iter} Adam steps, final loss {losses[-1]:.3f}")

    contacts_cpu = contacts.cpu()
    prec_main = [precision(top_pairs(s, L, MIN_SEP), contacts_cpu) for s in main_scores]
    prec_cov = [precision(top_pairs(s, L, MIN_SEP), contacts_cpu) for s in cov_scores]
    random_level = n_contacts / n_pairs
    for k in (L // 2, L):
        pk = precision(top_pairs(main_scores[-1], k, MIN_SEP), contacts_cpu)
        ck = precision(top_pairs(cov_scores[-1], k, MIN_SEP), contacts_cpu)
        print(f"top-{k:<3d} precision: {label} {pk:5.1%}   covariance {ck:5.1%}   random {random_level:5.1%}")
    print(f"total time {time.perf_counter() - t0:.1f} s")

    return plot.animate(
        seqs.cpu(), q, contacts_cpu, main_scores, cov_scores, xs, prec_main, prec_cov, random_level,
        label, params["method"] == "mfdca", MIN_SEP, out_dir / "coevolution_contacts.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
