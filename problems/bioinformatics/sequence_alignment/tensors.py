"""Build the concrete tensors: the DNA substitution matrix and a batch of related sequence pairs (synthetic, seeded)."""

import torch
import torch.nn.functional as F

ALPHABET = "ACGT"
PAD = 4  # code of the padding symbol after the end of a shorter sequence


def substitution_matrix(match: float, mismatch: float, device: torch.device) -> torch.Tensor:
    """W_{αβ}: `match` on the diagonal, `mismatch` everywhere else: (4, 4)."""
    eye = torch.eye(4, device=device)
    return match * eye + mismatch * (1 - eye)


def encode(seq: str) -> torch.Tensor:
    """'GATTACA' → tensor([2, 0, 3, 3, 0, 1, 0])."""
    return torch.tensor([ALPHABET.index(ch) for ch in seq])


def decode(codes: torch.Tensor) -> str:
    return "".join(ALPHABET[k] for k in codes.tolist() if k < 4)


def pad(seqs: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack sequences of different lengths: codes (B, L_max) padded with PAD, and lengths (B,)."""
    lengths = torch.tensor([len(s) for s in seqs])
    codes = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for b, s in enumerate(seqs):
        codes[b, : len(s)] = s
    return codes.to(device), lengths.to(device)


def one_hot(codes: torch.Tensor) -> torch.Tensor:
    """a_{biα}: (B, L) codes → (B, L, 4); padding becomes an all-zero row (it scores 0 and is never used)."""
    return F.one_hot(codes, 5)[..., :4].float()


def mutate(seq: torch.Tensor, rate: float, gen: torch.Generator) -> torch.Tensor:
    """Substitute a fraction `rate` of the letters and delete a fraction rate/4 of the sites."""
    flip = torch.rand(len(seq), generator=gen) < rate
    new = torch.randint(0, 4, (len(seq),), generator=gen)
    out = torch.where(flip, new, seq)
    keep = torch.rand(len(seq), generator=gen) >= rate / 4
    return out[keep]


def related_pairs(n_pairs: int, length: int, rate: float, seed: int, device: torch.device):
    """Global case: both sequences of a pair descend from one random ancestor (substitutions + indels).

    A deletion in one descendant is an insertion as seen from the other. Returns codes_a, len_a, codes_c, len_c.
    """
    gen = torch.Generator().manual_seed(seed)
    a, c = [], []
    for _ in range(n_pairs):
        ancestor = torch.randint(0, 4, (length,), generator=gen)
        a.append(mutate(ancestor, rate, gen))
        c.append(mutate(ancestor, rate, gen))
    return (*pad(a, device), *pad(c, device))


def planted_pairs(n_pairs: int, length: int, rate: float, seed: int, device: torch.device):
    """Local case: unrelated random sequences that share one planted motif (length/2) at random positions.

    Returns codes_a, len_a, codes_c, len_c and the planted spans (B, 4) = (start_a, end_a, start_c, end_c).
    """
    gen = torch.Generator().manual_seed(seed)
    a, c, spans = [], [], []
    k = max(6, length // 2)
    for _ in range(n_pairs):
        motif = torch.randint(0, 4, (k,), generator=gen)
        row = []
        for out in (a, c):
            copy = mutate(motif, rate, gen)
            start = int(torch.randint(0, length - k + 1, (1,), generator=gen))
            background = torch.randint(0, 4, (length - k,), generator=gen)
            out.append(torch.cat([background[:start], copy, background[start:]]))
            row += [start, start + len(copy)]
        spans.append(row)
    return (*pad(a, device), *pad(c, device), torch.tensor(spans))
