"""Build the concrete tensors: the 17th roots of unity, the cyclic ordering of the exponents and the level masks."""

import math

import torch

P = 17  # a Fermat prime: 17 = 2^(2^2) + 1, so (Z/17)^× has order 16 = 2^4
LEVELS = 5  # period lengths 16, 8, 4, 2, 1
FERMAT_PRIMES = (3, 5, 17, 257, 65537)  # the only known ones; their product is 2^32 − 1


def is_primitive_root(g: int, p: int = P) -> bool:
    """g generates (Z/p)^×: its powers g^0 … g^(p−2) are all different."""
    return len({pow(g, j, p) for j in range(p - 1)}) == p - 1


def cyclic_order(g: int, p: int = P) -> torch.Tensor:
    """The exponents in the order g^0, g^1, …, g^15 mod 17 (Gauss's ordering).   → (16,) int64"""
    if not is_primitive_root(g, p):
        raise ValueError(f"{g} is not a primitive root mod {p}")
    j = torch.arange(p - 1)
    return torch.tensor([pow(g, int(k), p) for k in j])


def roots_of_unity(p: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """ζ^k = e^{2πik/p} for k = 0 … p − 1.   → (p,) complex"""
    k = torch.arange(p, dtype=dtype, device=device)
    return torch.polar(torch.ones_like(k), 2 * math.pi * k / p)


def level_masks(device: torch.device) -> torch.Tensor:
    """M[m, i, j] = 1 when position j of the cyclic order belongs to period i at level m, i.e. j ≡ i mod 2^m.

    Level m has 2^m periods of 16 / 2^m terms; rows i ≥ 2^m are padding (all zero).   → (5, 16, 16)"""
    m = torch.arange(LEVELS, device=device)[:, None, None]
    i = torch.arange(P - 1, device=device)[None, :, None]
    j = torch.arange(P - 1, device=device)[None, None, :]
    return ((j % 2**m == i) & (i < 2**m)).double()
