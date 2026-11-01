"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import google, transition


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    d, n_iter = params["damping"], params["iterations"]
    walkers = 2000 if fast_mode() else params["walkers"]
    links, community, pos = tensors.web_graph(params["pages"], params["dangling"], params["seed"], dev)
    links = links.double()
    n = links.shape[0]
    s = transition(links)  # (N, N), column-stochastic
    print(f"web graph: {n} pages, {int(links.sum())} links, {int((links.sum(0) == 0).sum())} dangling pages")
    print(f"columns of S sum to 1: max |Σ_i S_ij − 1| = {(s.sum(0) - 1).abs().max().item():.1e}")

    ds = sorted({0.5, 0.7, 0.95, d})  # a batch of damping factors, solved together
    main = ds.index(d)
    g = google(s, torch.tensor(ds, dtype=s.dtype, device=dev))  # (B, N, N)
    history, residual = solve.power_iteration(g, n_iter)  # (K+1, B, N), (K, B)
    factor = solve.convergence_factor(residual)
    print("damping d   |λ₂| of G   observed rate ‖x_k+1 − x_k‖ / ‖x_k − x_k−1‖")
    lam2 = torch.stack([solve.eig_stationary(g[b])[1][1] for b in range(len(ds))])  # (B,)
    for b, db in enumerate(ds):
        print(f"   {db:.2f}      {lam2[b].item():.4f}      {factor[b].item():.4f}")

    pi = history[-1, main]
    pi_eig, _ = solve.eig_stationary(g[main])
    print(
        f"d = {d}: power iteration after {n_iter} steps vs eig: max difference {(pi - pi_eig).abs().max().item():.1e}"
    )
    print(f"PageRank sums to {pi.sum().item():.12f}, smallest entry {pi.min().item():.4f} (≥ (1 − d)/N)")

    naive = d * links / links.sum(0).clamp(min=1) + (1 - d) / n  # dangling columns left as zeros
    x = torch.full((n,), 1.0 / n, dtype=s.dtype, device=dev)
    for _ in range(n_iter):
        x = naive @ x
    print(f"without the dangling fix the total probability leaks away: Σ x = {x.sum().item():.4f} after {n_iter} steps")

    occupancy = solve.monte_carlo(s, d, walkers, n_iter, params["seed"])  # (K+1, N)
    estimate = occupancy[n_iter // 2 :].mean(dim=0).to(pi)  # time average once the surfers have mixed
    pearson = torch.corrcoef(torch.stack([estimate, pi]))[0, 1].item()
    print(f"{walkers} random surfers: L1 distance to PageRank {(estimate - pi).abs().sum().item():.4f}, ", end="")
    print(f"correlation {pearson:.4f}, rank correlation {solve.spearman(estimate, pi):.4f}")
    top = pi.argsort(descending=True)[:5].tolist()
    print("top pages (power):  " + "  ".join(f"#{i} {pi[i].item():.3f}" for i in top))
    top_mc = estimate.argsort(descending=True)[:5].tolist()
    print("top pages (surfers):" + "  ".join(f"#{i} {estimate[i].item():.3f}" for i in top_mc))

    return plot.animate(
        links,
        community,
        pos,
        history[:, main],
        pi_eig,
        occupancy,
        residual,
        ds,
        factor,
        lam2,
        d,
        out_dir / "pagerank.gif",
    )


if __name__ == "__main__":
    cli(run, __file__)
