"""Build the concrete tensors: the 2D grid and its wavenumbers, the barrier, the absorbing mask and the packet."""

import math

import torch


def grid(n: int, length: float, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Periodic (N, N) grid over [−L/2, L/2)². Returns (X, Y, KX, KY, dx); first index is x, second is y."""
    dx = length / n
    s = -length / 2 + dx * torch.arange(n, dtype=torch.float64, device=device)  # (N,)
    k = 2 * math.pi * torch.fft.fftfreq(n, d=dx, dtype=torch.float64, device=device)  # (N,) in FFT order
    X, Y = torch.meshgrid(s, s, indexing="ij")  # (N, N) each
    KX, KY = torch.meshgrid(k, k, indexing="ij")  # (N, N) each
    return X, Y, KX, KY, dx


def double_slit(X: torch.Tensor, Y: torch.Tensor, V0: float, sep: float, width: float, thick: float = 1.5):
    """V_{xy} = V0 inside a wall |x| < thick/2, except two openings |y ∓ sep/2| < width/2. Shape (N, N)."""
    wall = X.abs() < thick / 2
    slits = ((Y - sep / 2).abs() < width / 2) | ((Y + sep / 2).abs() < width / 2)
    return torch.where(wall & ~slits, V0, 0.0)


def disk(X: torch.Tensor, Y: torch.Tensor, V0: float, diameter: float) -> torch.Tensor:
    """V_{xy} = V0 inside a disk of the given diameter centred at the origin. Shape (N, N)."""
    return torch.where(X**2 + Y**2 < (diameter / 2) ** 2, V0, 0.0)


def absorbing_mask(X: torch.Tensor, Y: torch.Tensor, length: float, width: float) -> torch.Tensor:
    """M_{xy} = m(x) m(y): 1 in the middle, falling smoothly to 0 over `width` at every edge. Shape (N, N)."""

    def edge(s: torch.Tensor) -> torch.Tensor:
        depth = ((s.abs() - (length / 2 - width)) / width).clamp(0, 1)  # 0 inside, 1 at the edge
        return torch.cos(0.5 * math.pi * depth) ** 0.125  # gentle ramp: little reflection

    return edge(X) * edge(Y)


def gaussian_packet(X: torch.Tensor, Y: torch.Tensor, x0: float, sx: float, sy: float, k0: float, dA: float):
    """ψ_{xy} ∝ exp(−(x−x0)²/(4sx²) − y²/(4sy²) + i k0 x), moving along +x, Σ|ψ|² dA = 1. Shape (N, N) complex."""
    psi = torch.exp(-((X - x0) ** 2) / (4 * sx**2) - Y**2 / (4 * sy**2) + 1j * k0 * X)
    return psi / torch.sqrt((psi.abs() ** 2).sum() * dA)
