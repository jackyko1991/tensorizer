"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import standard_error


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    b, n = params["n_experiments"], params["n_darts"]
    x = tensors.darts(b, n, 0, dev)  # (B, N, 2)
    est = solve.darts_estimate(x)  # (B, N)
    sd = solve.spread(est)  # (N,)
    ns = torch.arange(1, n + 1, dtype=torch.float64, device=dev)

    print(f"{b} experiments × {n} darts")
    for k in (100, 1000, 10000, n):
        if k <= n:
            print(
                f"N = {k:7d}: mean π̂ = {est[:, k - 1].mean():.5f}   spread {sd[k - 1]:.5f}   "
                f"theory √(π(4−π)/N) = {standard_error(ns[k - 1]):.5f}"
            )
    final = est[:, -1]
    dev_from_pi = (final - math.pi).abs()
    print(f"best experiment {final[dev_from_pi.argmin()]:.6f}, worst {final[dev_from_pi.argmax()]:.6f}")

    y, phi = tensors.needles(b, n, 0, dev)
    needle = solve.needle_estimate(y, phi, params["needle_length"])  # (B,)
    print(f"Buffon's needle (ℓ/d = {params['needle_length']}): π̂ = {needle.mean():.5f} ± {needle.std():.5f}")
    print("4× the darts gives only 2× the accuracy: the error falls as 1/√N in every dimension")
    return plot.animate(x, est, sd, out_dir / "monte_carlo_pi.gif")


if __name__ == "__main__":
    cli(run, __file__)
