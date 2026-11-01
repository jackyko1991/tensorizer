"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import numpy as np

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import patristic


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    clock = params["clock"] == "strict"
    tree = tensors.random_tree(params["n_taxa"], params["tree_depth"], clock, seed=0)
    M, b = tensors.path_matrix(tree, dev)  # (n, V), (V,)
    d_true = patristic(M, b)  # (n, n) path lengths on the true tree
    X = tensors.simulate(tree, params["n_sites"], 1, seed=1, device=dev)[0]  # (n, L, 4)
    n, L = X.shape[0], X.shape[1]
    print(f"{n} taxa, {L} sites evolved under Jukes–Cantor along a {'clock-like' if clock else 'relaxed-clock'} tree")
    print(f"true tree: {len(solve.splits(tree))} inner branches, longest leaf-to-leaf path {float(d_true.max()):.2f}")

    n_frames = 4 if fast_mode() else 40
    sites = np.unique(np.geomspace(min(20, L), L, n_frames).astype(int))  # use the first `s` sites in each frame
    results = []
    for s in sites:
        p, d = solve.distances(X[:, :s])  # (n, n) each
        nj, up = solve.midpoint_root(solve.neighbour_joining(d)), solve.upgma(d)
        results.append((int(s), p, d, nj, up, solve.robinson_foulds(nj, tree), solve.robinson_foulds(up, tree)))
    for s, p, d, _, _, rf_nj, rf_up in results[:: max(1, len(results) // 5)] + results[-1:]:
        err_p = float((p - d_true).abs().mean())
        err_d = float((d - d_true).abs().mean())
        print(
            f"{s:5d} sites: mean |p − true| = {err_p:.3f}, |d_JC − true| = {err_d:.3f}, "
            f"RF(NJ) = {rf_nj}, RF(UPGMA) = {rf_up}"
        )
    worst = max(2 * n - 6, 1)
    print(f"Robinson–Foulds distance ranges from 0 (same unrooted shape) to {worst} for {n} taxa")
    return plot.animate(tree, results, d_true, out_dir / "phylogeny.gif")


if __name__ == "__main__":
    cli(run, __file__)
