"""Build the concrete tensors: random 3-SAT formulas, planted formulas and the table of all 2^n assignments."""

import torch


def random_3sat(n: int, m: int, batch: int, gen: torch.Generator, device: torch.device):
    """B random formulas with M clauses of 3 distinct variables, each negated with probability 1/2.

    var: (B, M, 3) int64 variable index, neg: (B, M, 3) bool (True ⇒ the literal is ¬x).
    The first M' clauses of a formula are themselves a random formula with M' clauses, so one formula of length
    M_max gives the whole sweep over the clause/variable ratio α = M'/n ("nested" formulas).
    """
    keys = torch.rand(batch, m, n, generator=gen)
    var = keys.argsort(dim=-1)[..., :3]  # 3 distinct variables per clause, uniformly
    neg = torch.rand(batch, m, 3, generator=gen) < 0.5
    return var.to(device), neg.to(device)


def planted_3sat(x_star: torch.Tensor, m: int, gen: torch.Generator):
    """Random formulas that the hidden assignment x_star (B, n) satisfies: each clause gets one of the 7 sign
    patterns that x_star satisfies (never the one that makes all 3 literals false)."""
    batch, n = x_star.shape
    keys = torch.rand(batch, m, n, generator=gen)
    var = keys.argsort(dim=-1)[..., :3].to(x_star.device)  # (B, M, 3)
    value = torch.gather(x_star[:, None].expand(-1, m, -1), 2, var)  # (B, M, 3) value of each variable
    pattern = torch.randint(1, 8, (batch, m), generator=gen).to(x_star.device)  # 1 … 7, never 0
    true_lit = (pattern[..., None] >> torch.arange(3, device=x_star.device)) & 1  # which literals x_star makes true
    neg = value ^ true_lit.bool()  # literal value = x XOR neg ⇒ choose neg so the chosen literals are true
    return var, neg


def all_assignments(n: int, device: torch.device) -> torch.Tensor:
    """Every assignment a = 0 … 2^n − 1 as bits, x_0 the most significant: (2^n, n) bool.
    In this order a depth-first search that tries x_i = 0 before 1 meets the leaves in increasing a."""
    a = torch.arange(2**n, device=device)
    shift = torch.arange(n - 1, -1, -1, device=device)
    return ((a[:, None] >> shift[None, :]) & 1).bool()


def truth_tables(n: int, device: torch.device) -> torch.Tensor:
    """The column x_j of all_assignments, packed 32 assignments per word: (n, W) int64 with W = 2^n / 32."""
    from .equation import pack

    return pack(all_assignments(n, device).T)
