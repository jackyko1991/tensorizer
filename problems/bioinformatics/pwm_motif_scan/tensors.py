"""Build the concrete tensors: a synthetic motif, aligned training sites, and a long sequence with planted sites.

Bases are numbered A = 0, C = 1, G = 2, T = 3, so the complement of base a is 3 - a.
"""

import torch

BASES = "ACGT"
IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "W": "AT", "R": "AG", "S": "CG"}
# A synthetic 12-bp site: a TATA-box-like core (TATAWAWR, W = A/T, R = A/G) with two preferred letters on each
# side. It is not a palindrome, so a site on the plus strand does not also score high on the minus strand.
CONSENSUS = "GCTATAWAWRAG"


def true_motif(strength: float, device: torch.device) -> torch.Tensor:
    """The hidden column probabilities M_{ja} used to generate sites.   (m, 4), rows sum to 1.

    In column j the letters allowed by the IUPAC code share the probability `strength`; the others share the rest.
    """
    rows = []
    for code in CONSENSUS:
        allowed = [a for a, b in enumerate(BASES) if b in IUPAC[code]]
        row = torch.full((4,), (1 - strength) / (4 - len(allowed)))
        row[allowed] = strength / len(allowed)
        rows.append(row)
    return torch.stack(rows).to(device)


def background(device: torch.device) -> torch.Tensor:
    """q_a: background letter frequencies of the random sequence (uniform here).   (4,)"""
    return torch.full((4,), 0.25, device=device)


def sample_sites(M: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """n aligned sites drawn column by column from M, as letter indices.   (n, m)"""
    return torch.multinomial(M.cpu(), n, replacement=True, generator=gen).T.to(M.device)


def one_hot(seq: torch.Tensor) -> torch.Tensor:
    """X_{…a} = 1 where the letter is a.   (…,) → (…, 4)"""
    return torch.nn.functional.one_hot(seq, num_classes=4).float()


def reverse_complement(seq: torch.Tensor) -> torch.Tensor:
    """Read the other strand: reverse the order and swap A↔T, C↔G (a → 3 - a).   (…, m) → (…, m)"""
    return 3 - seq.flip(-1)


def sequence_with_sites(
    M: torch.Tensor, length: int, n_sites: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A random sequence (L,) with n_sites motif instances; every second one is planted on the minus strand.

    Returns seq (L,), the start positions (n_sites,) and the strands (n_sites,) with +1 / -1.
    """
    m = M.shape[0]
    seq = torch.randint(0, 4, (length,), generator=gen).to(M.device)
    slot = length // n_sites  # one site somewhere inside each slot, so they never overlap
    starts = torch.arange(n_sites) * slot + torch.randint(0, slot - m, (n_sites,), generator=gen)
    strands = torch.tensor([1 if i % 2 == 0 else -1 for i in range(n_sites)])
    sites = sample_sites(M, n_sites, gen)  # (n_sites, m)
    for i in range(n_sites):
        site = sites[i] if strands[i] > 0 else reverse_complement(sites[i])
        seq[starts[i] : starts[i] + m] = site
    return seq, starts.to(M.device), strands.to(M.device)


def shuffles(seq: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """n random permutations of seq: same letter composition, no motif left.   (L,) → (n, L)"""
    order = torch.argsort(torch.rand(n, seq.shape[0], generator=gen), dim=1).to(seq.device)
    return seq[order]
