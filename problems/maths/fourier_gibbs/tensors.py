"""Build the concrete tensors: the sample grid and weights, the three waves, the sine basis, exact coefficients."""

import math

import torch

WAVES = ["square", "sawtooth", "triangle"]
JUMP = [2.0, 2.0, 0.0]  # size of the jump in each wave (the triangle is continuous)


def grid(n_x: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Midpoints x_j of n_x equal cells on (−π, π) and their widths w_j = h: (X,), (X,). No point sits on a jump."""
    h = 2 * math.pi / n_x
    x = -math.pi + h * (torch.arange(n_x, dtype=torch.float64, device=device) + 0.5)
    return x, torch.full_like(x, h)


def waves(x: torch.Tensor) -> torch.Tensor:
    """F_{wj}: the square, sawtooth and triangle waves at x_j. All are odd and swing between −1 and 1: (3, X)."""
    square = torch.sign(torch.sin(x))
    sawtooth = x / math.pi
    triangle = 2 / math.pi * torch.asin(torch.sin(x))
    return torch.stack([square, sawtooth, triangle])


def basis(x: torch.Tensor, n_max: int) -> torch.Tensor:
    """B_{jn} = sin(n x_j) for the harmonics n = 1 … N_max: (X, N_max). Odd waves need only sines."""
    n = torch.arange(1, n_max + 1, dtype=x.dtype, device=x.device)  # (N_max,)
    return torch.sin(x[:, None] * n[None, :])


def exact_coefficients(n_max: int, device: torch.device) -> torch.Tensor:
    """The textbook b_n of the three waves, to check the numerical integrals: (3, N_max)."""
    n = torch.arange(1, n_max + 1, dtype=torch.float64, device=device)
    odd = (n % 2 == 1).double()
    square = 4 / (math.pi * n) * odd  # 4/(πn) for odd n, 0 for even n
    sawtooth = 2 / (math.pi * n) * torch.where(n % 2 == 1, 1.0, -1.0)  # 2(−1)^{n+1}/(πn)
    triangle = 8 / (math.pi * n) ** 2 * torch.where(n % 4 == 1, 1.0, -1.0) * odd  # ±8/(π²n²) for odd n
    return torch.stack([square, sawtooth, triangle])
