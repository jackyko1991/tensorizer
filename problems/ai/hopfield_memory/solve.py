"""Recall with both networks: corrupted faces for a growing number of stored faces, and the capacity curves."""

import torch

from problems._common.progress import steps

from .equation import async_sweep, hebbian, modern_update, overlap


def classic_recall(
    patterns: torch.Tensor, used: torch.Tensor, queries: torch.Tensor, sweeps: int, seed: int
) -> torch.Tensor:
    """Run F classic networks (network f stores the patterns with used[f] = 1) from the same queries.
    (P, N), (F, P), (Q, N) → final states (F, Q, N)."""
    w = hebbian(patterns, used)  # (F, N, N)
    s = queries[None].expand(len(used), *queries.shape).clone()
    gen = torch.Generator().manual_seed(seed)
    for _ in steps(sweeps, "sweeps"):
        s = async_sweep(s, w, torch.randperm(patterns.shape[-1], generator=gen))
    return s


def modern_recall(patterns: torch.Tensor, used: torch.Tensor, queries: torch.Tensor, beta: float) -> torch.Tensor:
    """One modern update per network: only stored patterns take part (the others get −∞ similarity).
    (P, N), (F, P), (Q, N) → (F, Q, N)."""
    sim = beta * torch.einsum("qn,pn->qp", queries, patterns)  # (Q, P)
    sim = sim[None].masked_fill(used[:, None, :] == 0, float("-inf"))  # (F, Q, P)
    return torch.einsum("fqp,pn->fqn", torch.softmax(sim, dim=-1), patterns)


def capacity_curve(
    alphas: torch.Tensor, n: int, flip: float, n_queries: int, sweeps: int, beta: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean recall overlap of random patterns against the load α = P/N, from queries with a fraction `flip` of bits
    flipped: classic (A,), modern (A,), and the one-step bit error rate of the stored patterns (A,)."""
    loads = (alphas * n).round().long().clamp(min=n_queries)  # P for every α
    patterns = torch.randint(0, 2, (int(loads.max()), n), generator=torch.Generator().manual_seed(seed))
    patterns = (patterns * 2 - 1).float().to(alphas.device)  # (P_max, N)
    used = (torch.arange(len(patterns), device=alphas.device)[None] < loads[:, None]).float()  # (A, P_max)
    targets = patterns[:n_queries]  # (Q, N), stored in every network
    gen = torch.Generator().manual_seed(seed + 1)
    flips = (torch.rand(targets.shape, generator=gen) < flip).to(targets.device)
    queries = torch.where(flips, -targets, targets)

    classic = overlap(classic_recall(patterns, used, queries, sweeps, seed + 2), targets).mean(-1)
    modern = overlap(modern_recall(patterns, used, queries, beta), targets).mean(-1)
    w = hebbian(patterns, used)  # (A, N, N)
    h = torch.einsum("fij,pj->fpi", w, patterns)  # (A, P_max, N) one synchronous step from every pattern
    wrong = (torch.sign(h) != patterns[None]).float().mean(-1)  # (A, P_max)
    bit_error = (wrong * used).sum(-1) / used.sum(-1)
    return classic, modern, bit_error


def mixture(patterns: torch.Tensor, queries: torch.Tensor, beta: float, n_steps: int) -> torch.Tensor:
    """n_steps modern updates of every query: (P, N), (Q, N) → (Q, N)."""
    xi = queries
    for _ in range(n_steps):
        xi = modern_update(xi, patterns, beta)
    return xi
