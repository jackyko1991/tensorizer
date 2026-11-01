"""Build the concrete tensors: the periodic grid, the derivative matrices and the initial wave."""

import math

import torch

LENGTH = 2 * math.pi  # periodic domain [0, 2π)


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """N equally spaced points x_i on [0, 2π) and the spacing Δx."""
    dx = LENGTH / n
    return torch.arange(n, device=device, dtype=torch.float64) * dx, dx


def derivative_matrices(n: int, dx: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Periodic central differences as (N, N) matrices.

    D: (f_{i+1} - f_{i-1}) / (2Δx)       first derivative
    L: (f_{i+1} - 2 f_i + f_{i-1}) / Δx²  second derivative
    The index i ± 1 wraps around (torch.roll of the identity matrix).
    """
    eye = torch.eye(n, device=device, dtype=torch.float64)
    right = torch.roll(eye, 1, dims=1)  # right[i, i+1] = 1
    left = torch.roll(eye, -1, dims=1)  # left[i, i-1] = 1
    D = (right - left) / (2 * dx)
    L = (right - 2 * eye + left) / dx**2
    return D, L


def initial_wave(x: torch.Tensor, amplitude: float, offset: float) -> torch.Tensor:
    """u_0(x) = U_0 + A sin x: shape (N,). The steepest downhill slope is -A, so a shock forms at t* = 1/A."""
    return offset + amplitude * torch.sin(x)
