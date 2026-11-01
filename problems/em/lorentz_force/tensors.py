"""Build the concrete tensors: Levi-Civita symbol, uniform fields and the particle batch."""

import itertools

import torch

# (charge, mass) for up to six particles: opposite signs, double charge, heavier and lighter ones
SPECIES = [(1.0, 1.0), (-1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (-2.0, 1.0), (1.0, 0.5)]
EVEN = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]  # cyclic permutations of (x, y, z)


def levi_civita(device: torch.device) -> torch.Tensor:
    """ε_ijk of shape (3, 3, 3): +1 for even permutations, −1 for odd ones, 0 when an index repeats."""
    eps = torch.zeros(3, 3, 3, device=device)
    for i, j, k in itertools.permutations(range(3)):
        eps[i, j, k] = 1.0 if (i, j, k) in EVEN else -1.0
    return eps


def fields(E: float, B: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """E_i = (0, E, 0) and B_k = (0, 0, B): crossed fields, each of shape (3,)."""
    return torch.tensor([0.0, E, 0.0], device=device), torch.tensor([0.0, 0.0, B], device=device)


def particles(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Charges q_b and masses m_b of the first n species, each of shape (P,)."""
    q, m = zip(*SPECIES[:n])
    return torch.tensor(q, device=device), torch.tensor(m, device=device)


def initial_state(n: int, v_par: float, device: torch.device) -> torch.Tensor:
    """y0 of shape (P, 2, 3): every particle starts at the origin.

    y0[b, 0, :] = (0, 0, 0)          position x_{bi}
    y0[b, 1, :] = (0, 1, v_par)      velocity v_{bi}: unit speed ⊥ B plus a component along B
    """
    position = torch.zeros(n, 3, device=device)
    velocity = torch.tensor([0.0, 1.0, v_par], device=device).repeat(n, 1)  # (P, 3)
    return torch.stack([position, velocity], dim=1)
