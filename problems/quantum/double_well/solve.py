"""Diagonalise H once, fix the signs of the doublet, then evolve exactly in the eigenbasis."""

import math

import torch

from .equation import evolve


def eigenstates(H: torch.Tensor, x: torch.Tensor, dx: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (E_n (N,), φ_{xn} (N, N)) with ∫|φ|² dx = 1.

    eigh may return any eigenvector with either sign. We choose every state to be positive in the
    left well (summed over x < 0), so that φ_0 + φ_1 is the left-localised combination.
    """
    energies, phi = torch.linalg.eigh(H)
    phi = phi / math.sqrt(dx)
    left_weight = phi[x < 0].sum(dim=0)  # (N,)
    return energies, phi * torch.where(left_weight < 0, -1.0, 1.0)


def tunnelling_period(energies: torch.Tensor) -> float:
    """T = 2π / ΔE with ΔE = E_1 − E_0: time to tunnel to the right well and back."""
    return 2 * math.pi / (energies[1] - energies[0]).item()


def solve(
    phi: torch.Tensor, energies: torch.Tensor, c: torch.Tensor, t_max: float, n_steps: int = 240
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (times (T,), ψ (T, N) complex). No time stepping: every t is evaluated exactly."""
    times = torch.linspace(0.0, t_max, n_steps + 1, dtype=torch.float64)
    psi = torch.stack([evolve(phi, energies, c, t) for t in times.tolist()])  # (T, N)
    return times, psi
