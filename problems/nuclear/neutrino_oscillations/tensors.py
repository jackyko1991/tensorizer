"""Build the concrete tensors: squared masses, the baseline grid and a beam of sample neutrinos."""

import math

import torch

FLAVOURS = ["e", "mu", "tau"]


def squared_masses(dm21: float, dm31: float, device: torch.device) -> torch.Tensor:
    """m_i² in eV², normal ordering with the lightest set to 0 (only differences matter): shape (3,)."""
    return torch.tensor([0.0, dm21, dm31], dtype=torch.float64, device=device)


def baselines(l_min: float, l_max: float, n: int, device: torch.device) -> torch.Tensor:
    """L_k in km, log-spaced so short and long oscillations both show up: shape (K,)."""
    return torch.logspace(math.log10(l_min), math.log10(l_max), n, dtype=torch.float64, device=device)


def beam(n: int, l_min: float, l_max: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """n sample neutrinos for the picture: log-uniform positions (km), a lateral offset and a uniform number
    u_n that decides which flavour a detector at that position would see: three tensors of shape (n,)."""
    gen = torch.Generator().manual_seed(seed)
    log_l = math.log10(l_min) + (math.log10(l_max) - math.log10(l_min)) * torch.rand(n, generator=gen)
    return 10**log_l, torch.rand(n, generator=gen), torch.rand(n, generator=gen)
