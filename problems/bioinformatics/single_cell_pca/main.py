"""Main loop: synthetic counts → normalise → highly variable genes → PCA → k-means (→ t-SNE) → ARI → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import adjusted_rand_index, hvg_scores, normalise, pca

SEED = 0
N_INIT = 8  # k-means restarts, run side by side


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    t0 = time.perf_counter()
    n_cells = 300 if fast else params["n_cells"]

    # --- a synthetic experiment: K cell types, each with its own marker genes ---
    mu, markers = tensors.type_profiles(
        params["n_genes"], params["n_types"], params["n_markers"], params["fold_change"], SEED, dev
    )  # (K, G), (K, M)
    counts, labels, _ = tensors.sample_counts(n_cells, mu, params["dispersion"], params["depth_spread"], SEED + 1)
    lib = counts.sum(1)
    zeros = (counts == 0).float().mean().item()
    print(f"{n_cells} cells × {params['n_genes']} genes, {params['n_types']} cell types; library sizes "
          f"{lib.min():.0f} … {lib.max():.0f} (median {lib.median():.0f}); {zeros:.0%} zeros")  # fmt: skip

    # --- normalise, choose highly variable genes, PCA ---
    scaled, X = normalise(counts)  # (C, G)
    hvg = torch.argsort(hvg_scores(scaled), descending=True)[: params["n_hvg"]]  # (H,) gene indices
    hit = torch.isin(hvg, markers.flatten()).float().mean().item()
    print(f"highly variable genes: {len(hvg)} kept, {hit:.0%} of them are true marker genes")
    n_pcs = params["n_pcs"]
    y, _, ratio = pca(X[:, hvg], n_pcs)  # (C, P), explained-variance ratio of all components
    print("explained variance PC1…PC6: " + "  ".join(f"{r:.1%}" for r in ratio[:6].tolist()))
    # Cross-check with the randomised SVD used for big data (a few extra components help it converge).
    _, S_low, _ = torch.pca_lowrank(X[:, hvg], q=n_pcs + 10, niter=6)
    S_full = torch.linalg.svdvals(X[:, hvg] - X[:, hvg].mean(0))
    gap = ((S_low[:n_pcs] - S_full[:n_pcs]).abs().max() / S_full[0]).item()
    print(f"torch.pca_lowrank: top {n_pcs} singular values match the full SVD to {gap:.1e} (relative)")

    # --- k-means in the space of the first n_pcs components ---
    km = solve.kmeans(y, params["k"], N_INIT, 4 if fast else params["kmeans_iters"], SEED)
    best_inertia = km["inertia"][:, km["best"]]
    ari = [adjusted_rand_index(lab, labels) for lab in km["labels"]]
    finals = ", ".join(f"{v:.0f}" for v in km["inertia"][-1])
    print(f"k-means (k = {params['k']}, {N_INIT} restarts at once): inertia {best_inertia[0]:.0f} → "
          f"{best_inertia[-1]:.0f}; final inertias of the restarts {finals}")  # fmt: skip
    print(f"adjusted Rand index vs true types: {ari[0]:.3f} after the first assignment → {ari[-1]:.3f}")

    # --- a 2D picture: PC1–PC2, or a t-SNE map ---
    if params["embedding"] == "tsne":
        n_steps = 100 if fast else 300
        emb = solve.tsne(y, 30.0, n_steps, SEED)
        print(f"t-SNE: {n_steps} Adam steps on the PC scores (perplexity 30)")
    else:
        emb = y[:, :2]
    print(f"total time {time.perf_counter() - t0:.1f} s")

    return plot.animate(
        y.cpu(), emb.cpu(), labels.cpu(), km, ari, ratio.cpu(), n_pcs, params["embedding"],
        out_dir / "single_cell_pca.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
