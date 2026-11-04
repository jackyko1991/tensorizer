"""Cosmological parameters (Planck 2018 ΛCDM) and the grid in x = ln a.

Units: Mpc for comoving lengths, km/s for speeds, km/s/Mpc for H, so c/H is in Mpc.
"""

import math

import torch

C0 = 299_792.458  # speed of light today, km/s
H0 = 67.4  # Hubble constant, km/s/Mpc (Planck 2018)
OMEGA_M = 0.315  # matter
OMEGA_R = 9.1e-5  # radiation: photons + 3 massless neutrinos (Ω_r h² ≈ 4.15e-5)
OMEGA_L = 1 - OMEGA_M - OMEGA_R  # dark energy (flat universe)
A_REC = 1 / 1090  # scale factor at last scattering (z ≈ 1089)
STEP_WIDTH = 0.1  # e-folds over which c switches from its early to its present value


def log_a_grid(a_min: float, n: int, device: torch.device) -> torch.Tensor:
    """x = ln a from a_min to today (a = 1), uniform in x.   (A,)"""
    return torch.linspace(math.log(a_min), 0.0, n, dtype=torch.float64, device=device)


def sweep(final: float, n_frames: int, device: torch.device) -> torch.Tensor:
    """Frame-by-frame values 0 … final (the animation turns a model on gradually).   (F,)"""
    return torch.linspace(0.0, final, n_frames, dtype=torch.float64, device=device)
