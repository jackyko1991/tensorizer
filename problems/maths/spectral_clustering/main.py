"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import dirichlet_energy, gaussian_knn_weights, laplacian, squared_distances


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n = 150 if fast_mode() else params["n_points"]
    results = {}
    for name in tensors.DATASETS:
        x, truth = tensors.points(name, n, params["noise"], 0, dev)  # (N, 2), (N,)
        k = tensors.N_CLUSTERS[name]
        w = gaussian_knn_weights(squared_distances(x), params["sigma"], params["k_neighbours"])  # (N, N)
        lap = laplacian(w)
        evals_l, _ = solve.spectrum(lap)
        n_zero = solve.count_zero_eigenvalues(evals_l)
        n_comp = int(solve.connected_components(w).unique().numel())
        spectral, evals, embedding = solve.spectral_clustering(w, k, seed=0)
        plain = solve.kmeans(x, k, seed=0)
        ari_s, ari_k = solve.adjusted_rand_index(truth, spectral), solve.adjusted_rand_index(truth, plain)
        lam2, fiedler = solve.fiedler_vector(w)
        f = torch.randn(n, generator=torch.Generator().manual_seed(1), dtype=torch.float64).to(dev)
        print(f"{name}: N = {n}, {int((w > 0).sum()) // 2} edges, max |row sum of L| = {lap.sum(1).abs().max():.1e}")
        print(
            f"  smallest eigenvalue of L {evals_l[0]:+.1e} (≥ 0), zero eigenvalues {n_zero}, components {n_comp}; "
            f"f·Lf = {f @ lap @ f:.3f} = ½ΣW(fi−fj)² = {dirichlet_energy(w, f):.3f}"
        )
        print(f"  L_sym eigenvalues {[round(v, 4) for v in evals[: k + 2].tolist()]}, Fiedler value λ₂ = {lam2:.4f}")
        print(f"  adjusted Rand index: spectral {ari_s:.3f}, plain k-means {ari_k:.3f}")
        results[name] = {
            "x": x,
            "spectral": spectral,
            "kmeans": plain,
            "ari": (ari_s, ari_k),
            "evals": evals[:10],
            "fiedler": fiedler,
            "embedding": embedding,
            "w": w,
        }

    # vibration modes of a grid graph: L = L₁ ⊗ I + I ⊗ L₁ with eigenvalues (2 − 2cos πa/n) + (2 − 2cos πb/n)
    g = params["grid"]
    evals_g, modes = solve.spectrum(laplacian(tensors.grid_graph(g, dev)))
    a = torch.arange(g, dtype=torch.float64, device=dev)
    one_d = 2 - 2 * torch.cos(math.pi * a / g)
    exact = (one_d[:, None] + one_d[None, :]).flatten().sort().values
    print(f"grid {g}×{g}: max |λ − exact (2 − 2cos πa/n) + (2 − 2cos πb/n)| = {(evals_g - exact).abs().max():.1e}")
    n_modes = 12
    print(f"first grid eigenvalues {[round(v, 4) for v in evals_g[:6].tolist()]} (pairs = degenerate modes)")
    return plot.animate(results, modes[:, 1 : n_modes + 1], evals_g[1 : n_modes + 1], g, out_dir / "spectral.gif")


if __name__ == "__main__":
    cli(run, __file__)
