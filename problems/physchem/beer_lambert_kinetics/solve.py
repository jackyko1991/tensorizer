"""Simulate the experiment (kinetics → spectra + noise) and analyse it (spectra → concentrations)."""

import torch

from . import equation


def simulate(
    E: torch.Tensor, c0: torch.Tensor, t: torch.Tensor, k1: float, k2: float, path_length: float, noise: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the true concentrations c (T, S) and the measured spectra D (T, W) = l E c + noise."""
    K = equation.rate_matrix(k1, k2, c0.device, c0.dtype)
    c = equation.concentrations(K, c0, t)
    return c, equation.absorbance(E, c, path_length) + noise


def unmix(E: torch.Tensor, D: torch.Tensor, path_length: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Least-squares concentrations from the spectra.   Returns (ĉ (T, S), residual spectra (T, W), cond(E))."""
    c_hat = equation.recover(D, equation.pseudo_inverse(E), path_length)
    residual = D - equation.absorbance(E, c_hat, path_length)
    return c_hat, residual, torch.linalg.cond(E).item()
