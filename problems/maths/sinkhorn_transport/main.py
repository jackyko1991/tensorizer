"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import barycentric_map, cost_matrix, gaussian_w2_squared, w2_squared_1d

CHAIN = ["ring", "two blobs", "spiral", "ring"]


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    n = 60 if fast else params["n_points"]
    n_iter = 100 if fast else params["n_iter"]
    eps = tensors.epsilons(params["eps_min"], 1.0, 4 if fast else params["n_eps"], dev)  # (E,) from 1 down
    a = tensors.uniform_weights(n, dev)
    clouds = [tensors.SHAPES[name](n, k, dev) for k, name in enumerate(CHAIN[:-1])]
    clouds.append(clouds[0])

    legs = []
    for k in range(len(CHAIN) - 1):
        x, y = clouds[k], clouds[k + 1]
        c = cost_matrix(x, y)  # (N, N)
        out = solve.sinkhorn(a, a, c, eps, n_iter, eps_start=1.0)
        exact = c[torch.arange(n), solve.exact_assignment(c)].mean().item()  # Hungarian: the ε → 0 limit
        big = eps[eps >= 0.03]
        div = solve.sinkhorn_divergence(x, a, y, a, big, n_iter // 3)
        marg = (out["plan"].sum(-1) - a).abs().max() / a[0]
        name = f"{CHAIN[k]} → {CHAIN[k + 1]}"
        print(f"{name}: N = {n} points each, exact W2² (Hungarian) = {exact:.4f}")
        for e, cost, dual in zip(eps.tolist(), out["cost"].tolist(), out["dual"].tolist()):
            print(f"  ε = {e:7.4f}: <P, C> = {cost:.4f}   OT_ε = {dual:.4f}")
        print(
            f"  Sinkhorn divergence at ε = {big[-1]:.3f}: S_ε = {div[-1]:.4f} "
            f"(entropic bias removed; OT_ε there {out['dual'][len(big) - 1]:.4f})"
        )
        print(f"  largest relative marginal error {marg:.1e}; plan ≥ 0: {bool((out['plan'] >= 0).all())}")
        tx = barycentric_map(out["plan"][-1], y)  # where each source point goes at the smallest ε
        legs.append(
            {
                "x": x.cpu(),
                "y": y.cpu(),
                "tx": tx.cpu(),
                "plan": out["plan"].cpu(),
                "cost": out["cost"].cpu(),
                "dual": out["dual"].cpu(),
                "exact": exact,
                "div": (big.cpu(), div.cpu()),
                "name": name,
            }
        )

    x1, y1 = clouds[0][:, 0], clouds[1][:, 0]  # 1D check: the x-coordinates of ring and blobs
    c1 = cost_matrix(x1[:, None], y1[:, None])
    one_d = solve.sinkhorn(a, a, c1, eps[-1:], n_iter, eps_start=1.0)["cost"][0]
    print(
        f"1D (x-coordinates only): quantile formula W2² = {w2_squared_1d(x1, y1):.5f}, "
        f"Sinkhorn at ε = {eps[-1]:.4f}: {one_d:.5f}"
    )
    m1, m2 = torch.zeros(2, dtype=torch.float64), torch.tensor([1.0, 0.0], dtype=torch.float64)
    s1, s2 = torch.eye(2, dtype=torch.float64), torch.diag(torch.tensor([0.25, 2.0], dtype=torch.float64))
    print(
        "Gaussians N(0, I) → N((1, 0), diag(1/4, 2)): Bures closed form "
        f"W2² = {gaussian_w2_squared(m1, s1, m2, s2):.4f}"
    )
    return plot.animate(legs, eps.cpu(), out_dir / "sinkhorn.gif", 6 if fast else 22)


if __name__ == "__main__":
    cli(run, __file__)
