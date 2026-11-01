"""Build the concrete tensors: the grids, random initial conditions and the tiny flux-correction network."""

import math

import torch
from torch import nn

LENGTH = 2 * math.pi  # periodic domain [0, 2π)


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """Cell centres x_i on [0, 2π) and the cell width Δx."""
    dx = LENGTH / n
    return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * dx, dx


def random_initial(x: torch.Tensor, n: int, seed: int, n_modes: int = 4) -> torch.Tensor:
    """n random smooth waves u_0(x) = c + Σ_m a_m sin(m x + φ_m), m = 1…n_modes: shape (n, N).

    Amplitudes fall off as 1/m and are rescaled so max|u_0 - c| = 1; c is a small random mean. Fixed seed.
    """
    g = torch.Generator().manual_seed(seed)
    m = torch.arange(1, n_modes + 1, dtype=torch.float32)  # (M,)
    a = (2 * torch.rand(n, n_modes, generator=g) - 1) / m  # (n, M)
    phase = 2 * math.pi * torch.rand(n, n_modes, generator=g)
    c = 0.4 * (torch.rand(n, 1, generator=g) - 0.5)
    waves = torch.einsum("bm,bmx->bx", a, torch.sin(m[None, :, None] * x.cpu()[None, None, :] + phase[..., None]))
    waves = waves / waves.abs().amax(dim=1, keepdim=True)
    return (c + waves).to(x.device)


class FluxCorrection(nn.Module):
    """δF_{i+½} = CNN(u): three periodic 1D convolutions (stencil 5 cells), a few hundred weights.

    The last layer starts at zero, so before training the learned solver *is* the plain solver.
    """

    def __init__(self, channels: int = 16, kernel: int = 5):
        super().__init__()
        conv = dict(kernel_size=kernel, padding=kernel // 2, padding_mode="circular")
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, **conv),
            nn.GELU(),
            nn.Conv1d(channels, channels, **conv),
            nn.GELU(),
            nn.Conv1d(channels, 1, **conv),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, N) → δF: (B, N), entry i is the correction at face i+½."""
        return self.net(u[:, None, :])[:, 0, :]
