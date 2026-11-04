"""Build the concrete tensors: Shioda's characters of Fermat varieties, the units mod d and the lines on a Fermat
surface."""

import math

import torch


def characters(n: int, d: int) -> torch.Tensor:
    """All a = (a_0, …, a_{n+1}) with 1 ≤ a_i ≤ d − 1 and Σ a_i ≡ 0 mod d.   → (A, n + 2) int64

    Each one labels a one-dimensional piece V(a) of the primitive middle cohomology of the Fermat variety
    x_0^d + … + x_{n+1}^d = 0 (the eigenspaces of the group μ_d^{n+2} acting by roots of unity)."""
    digits = torch.arange(1, d, dtype=torch.int64)
    if d < 2:
        return torch.zeros(0, n + 2, dtype=torch.int64)
    grid = torch.cartesian_prod(*([digits] * (n + 2))).reshape(-1, n + 2)
    return grid[grid.sum(dim=1) % d == 0]


def units(d: int) -> torch.Tensor:
    """t with 1 ≤ t < d and gcd(t, d) = 1: the Galois group of ℚ(e^{2πi/d}).   → (T,)"""
    return torch.tensor([t for t in range(1, d) if math.gcd(t, d) == 1] or [1], dtype=torch.int64)


def fermat_lines(d: int) -> torch.Tensor:
    """The 3d² lines on x^d + y^d + z^d + w^d = 0, each spanned by two vectors of ℂ⁴.   → (3d², 2, 4) complex128

    For a pairing of the coordinates, e.g. {x, y}, {z, w}: x = α y, z = β w with α^d = β^d = −1."""
    roots = torch.exp(1j * math.pi * (2 * torch.arange(d, dtype=torch.float64) + 1) / d)  # d-th roots of −1
    alpha, beta = torch.meshgrid(roots, roots, indexing="ij")
    alpha, beta = alpha.flatten(), beta.flatten()  # (d²,)
    one, zero = torch.ones_like(alpha), torch.zeros_like(alpha)
    lines = []
    for (i, j), (k, m) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        v1 = torch.zeros(len(alpha), 4, dtype=torch.complex128)
        v2 = torch.zeros(len(alpha), 4, dtype=torch.complex128)
        v1[:, i], v1[:, j] = alpha, one  # x_i = α x_j
        v2[:, k], v2[:, m] = beta, one  # x_k = β x_m
        v1[:, k], v1[:, m], v2[:, i], v2[:, j] = zero, zero, zero, zero
        lines.append(torch.stack([v1, v2], dim=1))
    return torch.cat(lines)
