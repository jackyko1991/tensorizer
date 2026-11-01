"""Build the concrete tensors: periodic grid, wavenumbers, the dispersion propagator and the starting wave."""

import math

import torch

from .equation import bright_soliton, peregrine

BOX = 60.0
SEED = 0
SOLITON_SPEED = 1.0


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    """x (J,) on [−BOX/2, BOX/2), wavenumbers k (K = J,), and Δx."""
    dx = BOX / n
    x = -BOX / 2 + torch.arange(n, device=device, dtype=torch.float64) * dx
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx).to(device=device, dtype=torch.float64)
    return x, k, dx


def half_linear_step(k: torch.Tensor, dt: float) -> torch.Tensor:
    """e^{−i k² Δt / 4}: half a step of ψ̂_t = −i (k²/2) ψ̂.   (K,) complex"""
    return torch.exp(-0.25j * k**2 * dt)


def initial_wave(x: torch.Tensor, mode: str, noise: float, eta: float) -> torch.Tensor:
    """ψ0 (J,) complex128 for the chosen mode."""
    if mode == "peregrine":
        return peregrine(x, -5.0)  # rises to its peak at t = 5
    if mode == "soliton":
        return bright_soliton(x + 10.0, 0.0, eta, SOLITON_SPEED)  # starts at x = −10, moves right
    # modulation: background 1 plus a little complex noise (fixed seed, generated on the CPU so every device agrees)
    gen = torch.Generator().manual_seed(SEED)
    ripples = torch.randn(2, len(x), generator=gen, dtype=torch.float64).to(x.device)
    return 1.0 + noise * (ripples[0] + 1j * ripples[1])
