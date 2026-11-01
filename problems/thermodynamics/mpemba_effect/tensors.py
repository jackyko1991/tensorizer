"""Build the concrete tensors: the lattice potential, the neighbour matrix and the batch of initial states."""

import math

import torch

# Staircase potential, left to right: a raised plateau, a deep narrow well, a barrier, a shallow wide well.
SEGMENTS = [(16, 1.0), (4, -3.0), (2, None), (18, -2.0)]  # (number of sites, energy); None → the barrier height


def potential(barrier: float, device: torch.device) -> torch.Tensor:
    """U_i on N = 40 lattice sites, shape (N,), in units of k_B (so U/T is dimensionless)."""
    parts = [torch.full((n,), barrier if e is None else e, dtype=torch.float64) for n, e in SEGMENTS]
    return torch.cat(parts).to(device)


def left_basin(n_sites: int, device: torch.device) -> torch.Tensor:
    """Boolean mask (N,) of the sites left of the barrier (plateau + deep well)."""
    n_left = SEGMENTS[0][0] + SEGMENTS[1][0]
    return torch.arange(n_sites, device=device) < n_left


def neighbours(n_sites: int, device: torch.device) -> torch.Tensor:
    """A_ij = 1 if sites i and j are nearest neighbours on the open chain, else 0: shape (N, N)."""
    i = torch.arange(n_sites, device=device)
    return ((i[:, None] - i[None, :]).abs() == 1).double()


def boltzmann(u: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """p_bi = e^{-U_i / T_b} / Z_b for a batch of temperatures T (B,): shape (B, N). softmax keeps it finite."""
    return torch.softmax(-u[None, :] / T[:, None], dim=1)


def start_temperatures(T_bath: float, T_star: float, n: int, device: torch.device) -> torch.Tensor:
    """n initial temperatures from just above T_bath up to 30 T_bath (log spaced), plus the Mpemba point T*."""
    T = torch.logspace(math.log10(1.15 * T_bath), math.log10(30 * T_bath), n, dtype=torch.float64)
    T = torch.cat([T, torch.tensor([T_star], dtype=torch.float64)])
    return T.sort().values.to(device)
