"""Main loop: parameters → random formulas → exhaustive search sweep → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors

THRESHOLD = 4.267  # conjectured satisfiability threshold of random 3-SAT (Mézard–Parisi–Zecchina 2002)


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_max = 12 if fast_mode() else params["n_max"]
    batch = 60 if fast_mode() else params["instances"]
    gen = torch.Generator().manual_seed(params["seed"])
    n_values = [max(n_max - 8, 6), n_max - 4, n_max]  # n ≥ 5: at least one 32-bit word

    # 1. satisfiable fraction for every clause count M = 0 … 8n, for three sizes n
    p_sat, alpha = {}, {}
    for n in n_values:
        var, neg = tensors.random_3sat(n, 8 * n, batch, gen, dev)
        record = [round(a * n) for a in torch.arange(0, 8.01, 0.5).tolist()] if n == n_max else None
        res = solve.sweep(var, neg, n, record=record, keep=int(n == n_max))
        p_sat[n] = res["sat"].double().mean(dim=0).cpu()  # (8n + 1,)
        alpha[n] = torch.arange(8 * n + 1) / n
        print(
            f"n = {n:2d}: {batch} formulas × 2^{n} = {2**n:,} assignments each; P(sat) = 1/2 at α = M/n ≈ "
            f"{solve.crossing(p_sat[n], n):.2f} (90% → 10% window {solve.crossing(p_sat[n], n, 0.9):.2f} – "
            f"{solve.crossing(p_sat[n], n, 0.1):.2f})"
        )
        if n == n_max:
            maps, record_max = res["maps"].cpu(), record
            nodes = res["nodes"].double().median(dim=0).values.cpu()  # (len(record),)
    peak = record_max[int(nodes.argmax())] / n_max
    print(f"conjectured threshold for n → ∞: α_c ≈ {THRESHOLD} (finite n crosses higher and drifts down towards it)")
    print(
        f"backtracking search (n = {n_max}): median {nodes[0]:.0f} nodes at α = 0, peak {nodes.max():.0f} at "
        f"α ≈ {peak:.2f}, {nodes[-1]:.0f} at α = 8 (easy – hard – easy)"
    )

    # 2. the asymmetry behind P vs NP: checking a certificate versus searching all 2^n assignments
    sizes = list(range(6, n_max + 3, 2))
    cost = solve.timing(sizes, 64, gen, dev)
    for n, tv, ts in zip(cost["n"], cost["verify"], cost["search"]):
        print(f"n = {n:2d}: verify 64 certificates {tv * 1e3:7.3f} ms   search all 2^n assignments {ts * 1e3:9.2f} ms")
    return plot.animate(alpha, p_sat, maps, record_max, nodes, cost, n_max, THRESHOLD, out_dir / "p_vs_np_sat.gif")


if __name__ == "__main__":
    cli(run, __file__)
