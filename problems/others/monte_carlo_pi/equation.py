r"""Random numbers → an integral.

Monte Carlo replaces an integral by an average over random samples. A dart x uniform in the unit square lands
inside the quarter circle x_i x_i ≤ 1 with probability equal to its area, π/4, so

    \hat\pi_N = \frac{4}{N}\sum_{n=1}^{N} h_n,      h_n = [x_{ni} x_{ni} \le 1]      (i summed).

Each h_n is a coin flip with p = π/4, so the estimate is unbiased and its standard error is

    \sigma_N = 4\sqrt{p(1 - p)/N} = \sqrt{\pi(4 - \pi)/N} \propto 1/\sqrt N.

Buffon's needle (1777): a needle of length ℓ ≤ d dropped on lines a distance d apart crosses one with probability
2ℓ/(πd); its centre lies a distance y from the nearest line and it crosses when y ≤ (ℓ/2) sin φ.
"""

import math

import torch


def hits(x: torch.Tensor) -> torch.Tensor:
    r"""h_{bn} = [x_{bni} x_{bni} ≤ 1]   (i summed).   (B, N, 2) → (B, N)"""
    return (torch.einsum("bni,bni->bn", x, x) <= 1.0).double()


def running_estimate(h: torch.Tensor, scale: float) -> torch.Tensor:
    r"""\hat\pi_{bn} = scale · (h_{b1} + … + h_{bn}) / n for every prefix length n.   (B, N) → (B, N)"""
    n = torch.arange(1, h.shape[-1] + 1, dtype=h.dtype, device=h.device)
    return scale * torch.cumsum(h, dim=-1) / n


def standard_error(n: torch.Tensor) -> torch.Tensor:
    r"""σ_N = √(π(4 − π)/N) for the dart estimate."""
    return torch.sqrt(math.pi * (4 - math.pi) / n)


def needle_crosses(y: torch.Tensor, phi: torch.Tensor, length: float) -> torch.Tensor:
    r"""[y ≤ (ℓ/2) sin φ] with line spacing d = 1.   (B, N) → (B, N)"""
    return (y <= 0.5 * length * torch.sin(phi)).double()
