"""Build the concrete tensors: temperatures, a random spin lattice per temperature, checkerboard masks."""

import torch

from .equation import T_C


def temperatures(T: float, device: torch.device) -> torch.Tensor:
    """The chosen temperature plus two references: at T_c and well above it.  Shape (B,) = (3,)."""
    return torch.tensor([T, T_C, 3.5], device=device)


def random_spins(n_batch: int, L: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """s_{ij} = ±1 with equal chance ("infinitely hot" start).  Shape (B, 1, L, L): batch, channel, rows, cols."""
    torch.manual_seed(seed)  # fixed seed: the same picture every run
    return (2 * torch.randint(0, 2, (n_batch, 1, L, L)) - 1).float().to(device)


def checkerboard(L: int, device: torch.device) -> list[torch.Tensor]:
    """Two boolean masks of shape (L, L): sites with i + j even (black) and odd (white)."""
    i, j = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    black = ((i + j) % 2 == 0).to(device)
    return [black, ~black]
