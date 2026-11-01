"""Build the concrete tensors: the H/P sequence, the starting chains of every replica and the cooling schedule.

Lattice positions are integer (x, y) pairs, so a chain of N residues in R replicas is a (R, N, 2) tensor. It is
stored as float (whole numbers are exact in float32) so distances can use fast matrix products.
"""

import math

import torch

# Standard 2D HP benchmarks (Unger & Moult 1993) with their known ground-state energies.
SEQUENCES = {
    "benchmark_20": ("HPHPPHHPHPPHPHHPPHPH", -9),
    "benchmark_24": ("HHPPHPPHPPHPPHPPHPPHPPHH", -9),
    "benchmark_25": ("PPHPPHHPPPPHHPPPPHHPPPPHH", -8),
    "benchmark_36": ("PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", -14),
}


def sequence(name: str) -> tuple[str, int]:
    """The H/P string and its known lowest energy."""
    return SEQUENCES[name]


def hydrophobic(seq: str, device: torch.device) -> torch.Tensor:
    """h_i = 1 for H (hydrophobic) and 0 for P (polar): shape (N,)."""
    return torch.tensor([1.0 if c == "H" else 0.0 for c in seq], device=device)


def straight_chains(n_replicas: int, n: int, device: torch.device) -> torch.Tensor:
    """x_{rik}: every replica starts fully stretched along x, residue i at (i, 0): shape (R, N, 2)."""
    line = torch.stack([torch.arange(n, dtype=torch.float32), torch.zeros(n)], dim=-1)  # (N, 2)
    return line.expand(n_replicas, n, 2).clone().to(device)


def temperatures(t_start: float, t_end: float, n_steps: int, device: torch.device) -> torch.Tensor:
    """Simulated annealing: T_s falls geometrically from t_start to t_end (units of the contact energy ε/k_B): (S,)."""
    return torch.logspace(math.log10(t_start), math.log10(t_end), n_steps, device=device)
