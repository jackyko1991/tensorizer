"""Integrate the population balance with RK4, and a kinetic Monte Carlo of step growth for comparison."""

import torch

from problems._common.integrators import integrate
from problems._common.progress import steps

from .equation import rhs


def population_balance(P0: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor, s_max: float, dt: float):
    """Return (s (T,), P (T, 2, L)): both mechanisms advanced together from s = 0 to s_max = X_n,max - 1."""
    n_steps = max(1, int(round(s_max / dt)))
    dt = s_max / n_steps
    traj = integrate(lambda P, s: rhs(P, idx, mask), P0, dt, n_steps)
    s = torch.arange(n_steps + 1, dtype=P0.dtype, device=P0.device) * dt
    return s, traj


def kinetic_monte_carlo(n_monomers: int, events_at: list[int], L: int, seed: int) -> torch.Tensor:
    """Step growth one bond at a time: pick an ordered pair of distinct chains (the A end of one, the B end of
    the other) uniformly at random and join them. With equal reactivity every pair is equally likely, so this
    is an exact stochastic simulation of the population balance (time is not needed: after e events p = e/N_0).

    Returns chain counts per initial monomer, shape (len(events_at), L), at the requested numbers of events.
    """
    gen = torch.Generator().manual_seed(seed)
    total = max(events_at)
    u = torch.rand(total, 2, generator=gen, dtype=torch.float64).tolist()  # random numbers drawn up front
    chains = [1] * n_monomers  # chain lengths; order does not matter
    wanted = sorted(set(events_at))
    snaps = {}
    if 0 in wanted:
        snaps[0] = list(chains)
    for e in steps(total, "KMC bonds"):
        N = len(chains)
        i = int(u[e][0] * N)
        j = int(u[e][1] * (N - 1))
        j += j >= i  # a different chain
        chains[i] += chains[j]  # join j onto i ...
        chains[j] = chains[-1]  # ... and fill j's slot with the last chain
        chains.pop()
        if e + 1 in wanted:
            snaps[e + 1] = list(chains)
    counts = []
    for e in events_at:
        lengths = torch.tensor(snaps[e]).clamp_max(L + 1)  # anything longer than L lands in an overflow bin
        counts.append(torch.bincount(lengths, minlength=L + 2)[1 : L + 1].to(torch.float64) / n_monomers)
    return torch.stack(counts)
