"""Build the concrete tensors: natural frequencies, random initial phases, couplings and the ring."""

import math

import torch


def frequencies(distribution: str, n: int, width: float, device: torch.device, seed: int) -> torch.Tensor:
    """n natural frequencies ω_i with zero mean: Lorentzian of half-width γ or Gaussian of standard deviation σ.
    Stratified sampling: one random draw inside each of n equal-probability bins, u_i = (i + U_i)/n, then the inverse
    CDF. This keeps the finite-N sample close to the true density (fewer accidental clusters).   → (n,) float64"""
    gen = torch.Generator().manual_seed(seed)
    u = (
        torch.arange(n, dtype=torch.float64) + torch.rand(n, generator=gen, dtype=torch.float64).clamp(1e-3, 1 - 1e-3)
    ) / n
    if distribution == "lorentzian":
        omega = width * torch.tan(math.pi * (u - 0.5))  # inverse CDF of the Cauchy law
    else:
        omega = width * math.sqrt(2) * torch.erfinv(2 * u - 1)  # inverse CDF of the normal law
    return omega[torch.randperm(n, generator=gen)].to(device)


def initial_phases(batch: int, n: int, device: torch.device, seed: int) -> torch.Tensor:
    """Independent uniform phases in [0, 2π) for every batch member: (B, N) float64."""
    gen = torch.Generator().manual_seed(seed)
    return (2 * math.pi * torch.rand(batch, n, generator=gen, dtype=torch.float64)).to(device)


def couplings(k_max: float, count: int, device: torch.device) -> torch.Tensor:
    """count coupling strengths from 0 to k_max, one per batch member: (B, 1)."""
    return torch.linspace(0, k_max, count, dtype=torch.float64, device=device)[:, None]


def ring(n: int, device: torch.device) -> torch.Tensor:
    """Positions x_j = −π + 2πj/n on the ring: (n,) float64."""
    return -math.pi + 2 * math.pi * torch.arange(n, dtype=torch.float64, device=device) / n


def chimera_start(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Abrams and Strogatz's initial condition θ(x) = 6 ρ e^{−0.76 x²}, ρ uniform in [−½, ½]: a random bump.   (N,)"""
    gen = torch.Generator().manual_seed(seed)
    rho = torch.rand(x.shape, generator=gen, dtype=torch.float64).to(x.device) - 0.5
    return 6 * rho * torch.exp(-0.76 * x**2)
