"""Build the concrete tensors: the HMM parameters and synthetic DNA with planted CpG islands (seeded)."""

import torch
import torch.nn.functional as F

ALPHABET = "ACGT"
STATES = ("background", "island")


def model(stay_bg: float, stay_island: float, bg: list, island: list, device: torch.device):
    """Two-state HMM (π, A, E): start (2,), transitions (2, 2), emissions (2, 4) over A, C, G, T."""
    pi = torch.tensor([0.9, 0.1], dtype=torch.float64)  # float64: log-likelihoods of long sequences are large
    A = torch.tensor([[stay_bg, 1 - stay_bg], [1 - stay_island, stay_island]], dtype=torch.float64)
    E = torch.tensor([bg, island], dtype=torch.float64)
    E = E / E.sum(dim=1, keepdim=True)
    return pi.to(device), A.to(device), E.to(device)


def true_model(gc_island: float, device: torch.device):
    """Model that generates the data: background 40 % GC, islands `gc_island` GC, islands ~200 letters long."""
    at = (1 - gc_island) / 2
    return model(0.998, 0.995, [0.3, 0.2, 0.2, 0.3], [at, gc_island / 2, gc_island / 2, at], device)


def initial_guess(device: torch.device):
    """A vague starting point for Baum–Welch: the island state is only slightly GC-richer and switches too often."""
    return model(0.98, 0.95, [0.27, 0.23, 0.23, 0.27], [0.22, 0.28, 0.28, 0.22], device)


def planted_sequences(n_seq: int, length: int, n_islands: int, E: torch.Tensor, seed: int):
    """Hidden states z (B, T) with `n_islands` islands of 100–300 letters planted at random places, and letters
    x (B, T) drawn from the emission row of each position's state."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.zeros(n_seq, length, dtype=torch.long)
    for b in range(n_seq):
        for _ in range(n_islands):
            size = int(torch.randint(100, 301, (1,), generator=gen))
            start = int(torch.randint(0, length - size, (1,), generator=gen))
            z[b, start : start + size] = 1
    probs = E.cpu()[z]  # (B, T, 4): emission probabilities at every position
    x = torch.multinomial(probs.reshape(-1, 4), 1, generator=gen).reshape(n_seq, length)
    return x.to(E.device), z.to(E.device)


def one_hot(x: torch.Tensor) -> torch.Tensor:
    """x_{btα}: (B, T) letter codes → (B, T, 4)."""
    return F.one_hot(x, 4).double()
