"""Build the concrete tensors: the observation angles, the slit counts and the phasor angle shown in each frame."""

import math

import torch

FRAMES_PER_N = 6  # animation frames spent on each slit count


def sin_theta_grid(n_points: int = 4001) -> torch.Tensor:
    """Observation directions sin θ_t, fine enough to resolve the sharp peaks of 30 slits: (T,)."""
    return torch.linspace(-0.95, 0.95, n_points)


def slit_counts(n_max: int) -> list[int]:
    """N = 2, 3, …, n_max: one pattern per slit count."""
    return list(range(2, n_max + 1))


def phasor_angles(n_slits: int) -> torch.Tensor:
    """Phase step φ between neighbouring slits for the phasor panel, swept from the principal
    maximum (φ = 0) to the first zero (φ = 2π/N): shape (FRAMES_PER_N,)."""
    return torch.linspace(0.0, 2 * math.pi / n_slits, FRAMES_PER_N)
