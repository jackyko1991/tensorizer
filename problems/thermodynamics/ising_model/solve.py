"""Metropolis Monte Carlo with checkerboard updates: half the lattice flips in parallel, then the other half."""

import torch

from .equation import energy_per_spin, flip_cost, magnetization, neighbor_sum


def metropolis_sweep(s: torch.Tensor, T: torch.Tensor, masks: list[torch.Tensor], J: float = 1.0) -> torch.Tensor:
    """One sweep = one Metropolis attempt per site.   s: (B, 1, L, L), T: (B,)."""
    T = T[:, None, None, None]  # (B, 1, 1, 1)  one temperature per lattice in the batch
    for mask in masks:  # black sites, then white sites
        dE = flip_cost(s, neighbor_sum(s), J)  # (B, 1, L, L)
        accept = torch.rand_like(s) < torch.exp(-dE / T)  # always True when dE ≤ 0
        s = torch.where(mask & accept, -s, s)
    return s


def solve(s0: torch.Tensor, T: torch.Tensor, masks: list[torch.Tensor], n_sweeps: int, J: float = 1.0) -> dict:
    """Return snapshots (S+1, B, L, L) as int8, magnetisation and energy per spin (S+1, B)."""
    s = s0
    frames, m, E = [s0[:, 0].to(torch.int8)], [magnetization(s0)], [energy_per_spin(s0, J)]
    for _ in range(n_sweeps):
        s = metropolis_sweep(s, T, masks, J)
        frames.append(s[:, 0].to(torch.int8))
        m.append(magnetization(s))
        E.append(energy_per_spin(s, J))
    return {"spins": torch.stack(frames), "m": torch.stack(m), "E": torch.stack(E), "final": s}
