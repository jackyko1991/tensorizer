"""Diagonalise H once, then evolve the superposition exactly in the eigenbasis."""

import math

import torch

from .equation import evolve, project


def eigenstates(H: torch.Tensor, dx: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (E_n (N,), φ_{xn} (N, N)) sorted by energy, normalised so Σ_x φ_{xn}² dx = 1.

    eigh is for symmetric (Hermitian) matrices: real eigenvalues, orthonormal eigenvectors.
    Its vectors have Σ_x φ² = 1; dividing by √dx turns that into the continuum normalisation ∫|φ|² dx = 1.
    """
    energies, phi = torch.linalg.eigh(H)
    return energies, phi / math.sqrt(dx)


def coefficients(phi: torch.Tensor, psi0: torch.Tensor, dx: float) -> torch.Tensor:
    """c_n of the initial state ψ0 in the eigenbasis: (N,) complex."""
    return project(phi, psi0, dx)


def revival_time(L: float) -> float:
    """T = 2π / E_1 = 4L²/π: every phase e^{−i n² E_1 T} is back to 1."""
    return 4 * L**2 / math.pi


def solve(
    phi: torch.Tensor, energies: torch.Tensor, c: torch.Tensor, t_max: float, n_steps: int = 240
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), ψ (T, N) complex). No time stepping: every t is evaluated exactly."""
    times = torch.linspace(0.0, t_max, n_steps + 1, dtype=torch.float64)
    psi = torch.stack([evolve(phi, energies, c, t) for t in times.tolist()])  # (T, N)
    return times, psi
