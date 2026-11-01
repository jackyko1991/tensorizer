r"""Newton's method → tensor translation.

Every pixel p of the complex plane holds one starting point z_p, so the whole picture is one complex tensor.
A polynomial with coefficients c_k (lowest power first) is a contraction with the powers of z,

    p(z_p) = \sum_k c_k V_{pk},        V_{pk} = z_p^k        (k summed),

and its derivative has the shifted coefficients c'_k = (k + 1) c_{k+1}. Newton's step then updates every pixel at once,

    z_p \leftarrow z_p - p(z_p) / p'(z_p),

and the root a pixel has reached is the nearest one, j^*_p = \arg\min_j |z_p - r_j|.
"""

import torch


def powers(z: torch.Tensor, degree: int) -> torch.Tensor:
    r"""V_{pk} = z_p^k for k = 0..degree, built by a running product (exact, unlike z ** k).   (...) → (..., K)"""
    ones = torch.ones_like(z)[..., None]
    return torch.cumprod(torch.cat([ones, z[..., None].expand(*z.shape, degree)], dim=-1), dim=-1)


def poly(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    r"""p(z_p) = c_k V_{pk}   (k summed).   z: (...), c: (K,) → (...)"""
    return torch.einsum("...k,k->...", powers(z, len(c) - 1), c)


def derivative(c: torch.Tensor) -> torch.Tensor:
    r"""c'_k = (k + 1) c_{k+1}: coefficients of p'(z).   (K,) → (K-1,)"""
    k = torch.arange(1, len(c), device=c.device)
    return k * c[1:]


def newton_step(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    r"""z ← z − p(z)/p'(z) on every pixel at once.   (...) → (...)"""
    return z - poly(z, c) / poly(z, derivative(c))


def nearest_root(z: torch.Tensor, roots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Distance to the nearest root min_j |z_p − r_j| and its index j^*_p.   z: (...), roots: (J,) → (...), (...)"""
    dist = (z[..., None] - roots).abs()  # (..., J)
    return dist.min(dim=-1)
