"""Exhaustive search over all 2^n assignments, clause by clause, for a whole batch of nested formulas at once."""

import math

import torch

from problems._common.progress import steps

from .equation import MASK, WORD, clause_table, unpack
from .tensors import truth_tables


def solution_table(var: torch.Tensor, neg: torch.Tensor, n: int) -> torch.Tensor:
    """Packed truth table of each whole formula: bit a is set iff assignment a satisfies it.   → (B, W)"""
    tables = truth_tables(n, var.device)  # (n, W)
    table = torch.full((var.shape[0], tables.shape[1]), MASK, dtype=torch.int64, device=var.device)
    for m in range(var.shape[1]):  # AND in one clause at a time (all formulas at once)
        table &= clause_table(tables, var[:, m], neg[:, m])
    return table


def first_solution(table: torch.Tensor, n: int) -> torch.Tensor:
    """Smallest satisfying assignment index a* per formula, 2^n if there is none.   (B, W) → (B,)"""
    bits = unpack(table, n)
    return torch.where(bits.any(dim=1), bits.long().argmax(dim=1), 2**n)


def backtrack_nodes(levels: torch.Tensor, n: int) -> torch.Tensor:
    """Nodes a depth-first backtracking search visits before it finds the first solution (or proves there is none).

    levels[:, d] is the packed table of the clauses whose variables all lie among x_0 … x_{d−1}: the partial
    assignments of depth d that no fully assigned clause rejects. The search fixes x_0, x_1, … in turn, tries 0 before
    1 and backs up as soon as a clause is violated, so it meets the leaves in increasing order. A live node of depth d
    covers a block of 2^(n−d) leaves; it is visited iff its block starts at or before the first solution a*.
    levels: (B, n + 1, W) → (B,) int64
    """
    a_star = first_solution(levels[:, n], n).clamp(max=2**n - 1)  # unsatisfiable ⇒ the whole tree is searched
    nodes = torch.zeros(levels.shape[0], dtype=torch.int64, device=levels.device)
    for d in range(n + 1):
        block = 2 ** (n - d)
        if block >= WORD:  # a block starts at bit 0 of every (block / 32)-th word
            live = (levels[:, d, :: block // WORD] & 1).bool()  # (B, 2^d): one entry per node of depth d
        else:
            live = unpack(levels[:, d], n)[:, ::block]
        start = torch.arange(2**d, device=levels.device) * block
        nodes += (live & (start[None, :] <= a_star[:, None])).sum(dim=1)
    return nodes


def sweep(var: torch.Tensor, neg: torch.Tensor, n: int, record: list[int] | None = None, keep: int = 0) -> dict:
    """Add the clauses of nested formulas one by one (M' = 0 … M_max) and track, for every prefix length M':

    sat (B, M_max + 1) bool       is the formula with the first M' clauses satisfiable?
    count (B, M_max + 1) int64    number of satisfying assignments (only for the first `keep` formulas)
    nodes (B, len(record)) int64  backtracking search cost at the prefix lengths listed in `record`
    maps (M_max + 1, 2^n) bool    solution map of formula 0 (only if keep > 0)
    """
    record = record or []
    batch, m_max, _ = var.shape
    tables = truth_tables(n, var.device)  # (n, W)
    levels = torch.full((batch, n + 1, tables.shape[1]), MASK, dtype=torch.int64, device=var.device)
    depth = torch.arange(n + 1, device=var.device)
    sat = torch.ones(batch, m_max + 1, dtype=torch.bool, device=var.device)
    maps = [unpack(levels[0, n], n)] if keep else []
    nodes = []
    if 0 in record:
        nodes.append(backtrack_nodes(levels, n))
    for m in steps(m_max, "clauses"):
        clause = clause_table(tables, var[:, m], neg[:, m])  # (B, W)
        deeper = depth[None, :] > var[:, m].max(dim=1).values[:, None]  # (B, n + 1): levels that see this clause
        levels = torch.where(deeper[..., None], levels & clause[:, None], levels)
        sat[:, m + 1] = (levels[:, n] != 0).any(dim=1)
        if keep:
            maps.append(unpack(levels[0, n], n))
        if m + 1 in record:
            nodes.append(backtrack_nodes(levels, n))
    out = {"sat": sat}
    if record:
        out["nodes"] = torch.stack(nodes, dim=1)
    if keep:
        out["maps"] = torch.stack(maps)
    return out


def crossing(p_sat: torch.Tensor, n: int, level: float = 0.5) -> float:
    """Ratio α = M/n where the (non-increasing) satisfiable fraction first drops below `level`, linearly
    interpolated between neighbouring clause counts."""
    hits = torch.nonzero(p_sat < level)
    if len(hits) == 0:
        return float("nan")  # never drops below `level` within the sweep
    below = int(hits[0])
    p0, p1 = p_sat[below - 1].item(), p_sat[below].item()
    m = below - 1 + (p0 - level) / max(p0 - p1, 1e-12)
    return m / n


def timing(n_values: list[int], batch: int, gen: torch.Generator, device: torch.device) -> dict:
    """Wall time to verify a batch of certificates versus to search all 2^n assignments, at α ≈ 4.27."""
    import time

    from .equation import verify
    from .tensors import random_3sat

    out = {"n": n_values, "verify": [], "search": []}
    var, neg = random_3sat(6, 26, batch, gen, device)
    verify(var[..., 0] > 2, var, neg), solution_table(var, neg, 6)  # warm-up: first calls pay one-off costs
    for n in n_values:
        m = math.ceil(4.27 * n)
        var, neg = random_3sat(n, m, batch, gen, device)
        x = torch.rand(batch, n, generator=gen).to(device) < 0.5
        t0 = time.perf_counter()
        for _ in range(20):
            verify(x, var, neg)
        t1 = time.perf_counter()
        for _ in range(3):
            solution_table(var, neg, n)
        t2 = time.perf_counter()
        out["verify"].append((t1 - t0) / 20)
        out["search"].append((t2 - t1) / 3)
    return out
