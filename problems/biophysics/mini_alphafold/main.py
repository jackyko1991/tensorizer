"""Main loop: target fold → predicted distogram → (attention refinement) → structure module → alignment → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import classical_mds, distogram, distogram_mean_var, pair_distances, rmsd, tm_score

D_FAR = 18.0  # Å: beyond this the binned mean is biased (the distogram ends at 22 Å), so only say "far apart"


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    target = tensors.target_structure(params["helix_length"], dev)  # (N, 3) Å
    n = target.shape[0]
    D = pair_distances(target)  # (N, N) true distances
    print(f"target: helix–turn–helix, {n} residues (Cα only), radius of gyration {_rg(target):.1f} Å")

    # --- step 1: the network's job, a distogram ---
    d_noisy, sigma = tensors.noisy_distances(D, params["noise"])
    centres = tensors.bin_centres(dev)  # (K,)
    p = distogram(d_noisy, sigma, centres)  # (N, N, K)
    d_hat, var = distogram_mean_var(p, centres)  # (N, N), (N, N)
    w = 1.0 / (var + 0.1)  # confident pairs count more
    far = d_hat > D_FAR
    near = ~far & ~torch.eye(n, dtype=torch.bool, device=dev)
    print(f"distogram: {len(centres)} bins 2–22 Å; {far.sum().item() // 2} pairs only known as '≥ {D_FAR:.0f} Å'")
    print(f"  mean |d̂ - D| over near pairs: {(d_hat - D)[near].abs().mean():.2f} Å (noise level {params['noise']})")

    attn = None
    if params["refine"] == "on":
        feats = tensors.positional_features(n, 32, dev)  # (N, E)
        d_used, attn = solve.refine(d_hat, w, far, D_FAR, feats, 60 if fast else 400)
        print(f"  after attention refinement:  {(d_used - D)[near].abs().mean():.2f} Å")
    else:
        d_used = d_hat

    # --- step 2: the structure module ---
    x0 = solve.initial_guesses(d_used, params["starts"])  # (B, N, 3), start 0 = classical MDS
    mds = classical_mds(d_used)
    print(
        f"classical MDS alone: RMSD {rmsd(mds, target):.2f} Å, its mirror image {rmsd(solve.mirror(mds), target):.2f} Å"
    )
    n_steps = 80 if fast else params["steps"]
    at, snaps, history, final = solve.fold(x0, d_used, w, far, D_FAR, 10.0, n_steps)  # (T, B, N, 3)

    best = int(final.argmin())
    model, flipped = solve.choose_hand(snaps[-1, best])
    rms_all = solve.best_rmsd_any_hand(snaps[-1], target)  # (B,)
    print(f"gradient descent from {x0.shape[0]} starts ({n_steps} Adam steps): final stress {final.min():.1f} … ")
    print(f"  {final.max():.1f}; RMSD (best hand) from {rms_all.min():.2f} to {rms_all.max():.2f} Å")
    print(f"  lowest stress: start {best} ({'MDS' if best == 0 else 'random'}); mirror flipped: {flipped}")
    r_model, r_mirror = rmsd(model, target).item(), rmsd(solve.mirror(model), target).item()
    print(f"  chosen hand: RMSD {r_model:.2f} Å (the other enantiomer: {r_mirror:.2f} Å)")
    print(f"  TM-like score {tm_score(model, target):.3f}  (> 0.5 means the same fold)")

    # animate the best *random* start, so the fold visibly assembles from a cloud
    show = 1 + int(final[1:].argmin()) if x0.shape[0] > 1 else 0
    traj = snaps[:, show]
    if solve.choose_hand(traj[-1])[1]:
        traj = solve.mirror(traj)
    rms_traj = rmsd(traj, target)  # (T,)
    rms_starts = solve.best_rmsd_any_hand(snaps, target)  # (T, B)
    return plot.animate(target, traj, d_used, far, attn, rms_traj, rms_starts, at, out_dir / "mini_alphafold.gif")


def _rg(x: torch.Tensor) -> float:
    return torch.sqrt(((x - x.mean(0)) ** 2).sum(1).mean()).item()


if __name__ == "__main__":
    cli(run, __file__)
