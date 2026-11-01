"""Evaluate every process at once: temperature, work, internal energy and heat."""

import torch

from .equation import heat, internal_energy_change, running_work, temperature, work


def solve(paths: torch.Tensor, n: float, gamma: float) -> dict[str, torch.Tensor]:
    """paths: (4, K, 2) samples of (V, p).  Returns totals (4,) and running totals (4, K), in joules."""
    V, p = paths[..., 0], paths[..., 1]  # (4, K) each

    T = temperature(V, p, n)  # (4, K)
    W_run = running_work(V, p)  # (4, K)  area under the curve so far
    dU_run = internal_energy_change(T, n, gamma)  # (4, K)
    Q_run = heat(dU_run, W_run)  # (4, K)

    return {
        "T": T,
        "W": work(V, p),  # (4,)  same as W_run[:, -1]
        "dU": dU_run[:, -1],
        "Q": Q_run[:, -1],
        "W_run": W_run,
        "dU_run": dU_run,
        "Q_run": Q_run,
    }
