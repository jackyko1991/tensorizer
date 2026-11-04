r"""Mathematics → tensor translation.

A 3-SAT formula is an AND of M clauses, each an OR of 3 literals: F(x) = ∧_m ∨_k ℓ_{mk}(x), with the literal
ℓ_{mk}(x) = x_{v_{mk}} ⊕ s_{mk} (variable v, negation sign s).

1. Verifying a certificate (the "NP" in P vs NP): given x, evaluate every literal at once with one gather, then
   OR over k and AND over m. Cost O(M): polynomial.

       ℓ_{bmk} = x_{b, v_{bmk}} ⊕ s_{bmk},   F_b = ∧_m ∨_k ℓ_{bmk}

2. Exhaustive search: evaluate F at all 2^n assignments at once. The 2^n values of x_j form a bit vector (its truth
   table T_j); a clause's table is T_{v1}⊕s1 | T_{v2}⊕s2 | T_{v3}⊕s3 and the formula's table is the AND of the
   clause tables. Packing 32 assignments per integer makes this bitwise tensor algebra, but the tensor still has
   2^n bits: the cost doubles with every extra variable.
"""

import torch

WORD = 32
MASK = (1 << WORD) - 1  # all 32 low bits set: "true for every assignment in this word"


def literals(x: torch.Tensor, var: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    r"""ℓ = x_v ⊕ s for every literal.   x: (..., n) bool, var/neg: (..., M, 3) → (..., M, 3) bool"""
    batch = torch.broadcast_shapes(x.shape[:-1], var.shape[:-2])
    m = var.shape[-2]
    x = x.expand(*batch, x.shape[-1])
    idx = var.flatten(-2).expand(*batch, 3 * m)
    return torch.gather(x, -1, idx).view(*batch, m, 3) ^ neg


def verify(x: torch.Tensor, var: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    r"""F(x) = ∧_m ∨_k ℓ_{mk}: is x a satisfying assignment?   → (...,) bool. Polynomial: 3M literal lookups."""
    return literals(x, var, neg).any(dim=-1).all(dim=-1)


def pack(bits: torch.Tensor) -> torch.Tensor:
    """(..., 2^n) bool → (..., 2^n / 32) int64; assignment a sits in word a // 32 at bit a % 32."""
    b = bits.reshape(*bits.shape[:-1], -1, WORD).long()
    return (b << torch.arange(WORD, device=bits.device)).sum(dim=-1)  # distinct powers of two: sum = OR


def unpack(words: torch.Tensor, n: int) -> torch.Tensor:
    """(..., W) int64 → (..., 2^n) bool."""
    bits = (words[..., None] >> torch.arange(WORD, device=words.device)) & 1
    return bits.reshape(*words.shape[:-1], 2**n).bool()


def popcount(words: torch.Tensor) -> torch.Tensor:
    """Number of set bits = number of satisfying assignments.   (..., W) → (...,)"""
    return ((words[..., None] >> torch.arange(WORD, device=words.device)) & 1).sum(dim=(-1, -2))


def clause_table(tables: torch.Tensor, var: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    r"""Truth table of one clause per formula: T_{v1}⊕s1 | T_{v2}⊕s2 | T_{v3}⊕s3.
    tables: (n, W), var/neg: (B, 3) → (B, W) int64"""
    lit = tables[var] ^ (neg.long() * MASK)[..., None]  # (B, 3, W): XOR with all ones negates
    return lit[:, 0] | lit[:, 1] | lit[:, 2]
