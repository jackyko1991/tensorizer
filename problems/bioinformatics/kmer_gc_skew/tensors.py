"""Build the concrete tensors: a synthetic circular genome, its one-hot encoding and the k-mer place values.

Bases are numbered A = 0, C = 1, G = 2, T = 3, so the complement of base a is 3 - a.
"""

import torch

BASES = "ACGT"
ISLANDS = [(0.62, -0.12), (0.85, -0.10)]  # (start as a fraction of the genome, change of GC content)


def synthetic_genome(
    length: int, origin_frac: float, gc: float, skew: float, seed: int, device: torch.device
) -> tuple[torch.Tensor, int, int]:
    """A random circular chromosome with a planted replication origin and terminus.

    From the origin clockwise to the terminus (half the circle) is the leading strand, where G is more common than C
    by the factor (1 + skew) / (1 - skew); on the other half C and G swap. Two short "islands" have a lower GC content.
    Returns seq (L,) with base indices 0..3, the origin position and the terminus position.
    """
    origin = int(origin_frac * length) % length
    terminus = (origin + length // 2) % length
    pos = torch.arange(length)
    leading = (pos - origin) % length < length // 2  # (L,) True on the half from origin to terminus

    gc_n = torch.full((length,), gc)  # (L,) GC content at every position
    for start, delta in ISLANDS:
        a = int(start * length)
        gc_n[a : a + length // 25] += delta
    s = torch.where(leading, skew, -skew)  # (L,) +skew on the leading half, -skew on the lagging half

    # probabilities of A, C, G, T at each position: (L, 4)
    prob = torch.stack([(1 - gc_n) / 2, gc_n / 2 * (1 - s), gc_n / 2 * (1 + s), (1 - gc_n) / 2], dim=1)
    u = torch.rand(length, 1, generator=torch.Generator().manual_seed(seed))
    seq = (u > prob.cumsum(dim=1)).sum(dim=1).clamp(max=3)  # inverse-CDF sampling: one base per position
    return seq.to(device), origin, terminus


def one_hot(seq: torch.Tensor) -> torch.Tensor:
    """X_{na} = 1 if base n is letter a, else 0.   (L,) → (L, 4)."""
    return torch.nn.functional.one_hot(seq, num_classes=4).float()


def place_values(k: int, device: torch.device) -> torch.Tensor:
    """P_{ja} = a · 4^(k-1-j): reading a k-mer as a number in base 4 (AAA = 0, AAC = 1, …, TTT = 4^k - 1).   (k, 4)"""
    digit = torch.arange(4, device=device, dtype=torch.float32)  # value of letter a
    weight = 4.0 ** torch.arange(k - 1, -1, -1, device=device, dtype=torch.float32)  # 4^(k-1-j)
    return torch.einsum("j,a->ja", weight, digit)


def kmer_names(k: int) -> list[str]:
    """All 4^k words in the same order as the k-mer index."""
    names = [""]
    for _ in range(k):
        names = [w + b for w in names for b in BASES]
    return names


def gc_strand_weights(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """g_a = [a ∈ {G, C}] and c_a = [a = G] - [a = C]: the per-base GC indicator and the skew sign.   (4,), (4,)"""
    is_gc = torch.tensor([0.0, 1.0, 1.0, 0.0], device=device)
    g_minus_c = torch.tensor([0.0, -1.0, 1.0, 0.0], device=device)
    return is_gc, g_minus_c
