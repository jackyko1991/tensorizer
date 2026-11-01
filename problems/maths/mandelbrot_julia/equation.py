r"""Mathematics → tensor translation.

Every pixel p of the picture is a complex number. One step of the quadratic map, for all pixels at once, is

    z_p ← z_p^2 + c_p,        z'_p ← 2 z_p z'_p + c'_p.

Mandelbrot set M:  c_p = pixel, z_0 = 0, the derivative is taken with respect to c, so c' = 1 and z'_0 = 0.
Julia set J_c:     c fixed,     z_0 = pixel, the derivative is taken with respect to z_0, so c' = 0 and z'_0 = 1.

No index is summed: the pixel index p only runs along, which is why the whole picture is one tensor operation.
A point escapes once |z_n| > R; if |z| ever exceeds 2 (and |c| ≤ 2) it is certain to run off to infinity.

Smooth colouring:   ν = n + 1 − log_2(ln|z_n|)          (the integer count n made continuous)
Distance estimate:  d = |z_n| ln|z_n| / |z'_n|           (≈ distance from the pixel to the set's boundary)
"""

import torch


def step(z: torch.Tensor, dz: torch.Tensor, c: torch.Tensor, dc: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""z ← z² + c and z' ← 2 z z' + c' for every pixel.   All complex tensors of the same shape (P,)."""
    return z * z + c, 2 * z * dz + dc


def smooth_count(n: torch.Tensor, z_abs: torch.Tensor) -> torch.Tensor:
    r"""ν = n + 1 − log₂(ln|z_n|): continuous escape count, so colour bands do not show steps.   (P,) → (P,)"""
    return n + 1 - torch.log2(torch.log(z_abs))


def distance_estimate(z_abs: torch.Tensor, dz_abs: torch.Tensor) -> torch.Tensor:
    r"""d = |z| ln|z| / |z'|: distance from an escaping point to the boundary, within about a factor 2.   (P,)"""
    return z_abs * torch.log(z_abs) / dz_abs


def in_cardioid_or_bulb(c: torch.Tensor) -> torch.Tensor:
    r"""Exact tests for the two biggest pieces of M (no iteration needed).

    Main cardioid: c = w/2 − w²/4 with |w| ≤ 1, where z* = w/2 is an attracting fixed point (|2 z*| ≤ 1).
    In coordinates, q = (x − ¼)² + y² and q (q + x − ¼) ≤ y²/4.
    Period-2 bulb: z → z² + c has an attracting 2-cycle iff |4(1 + c)| ≤ 1, the disc |c + 1| ≤ ¼.
    """
    x, y = c.real, c.imag
    q = (x - 0.25) ** 2 + y**2
    cardioid = q * (q + x - 0.25) <= 0.25 * y**2
    bulb = (x + 1) ** 2 + y**2 <= 1 / 16
    return cardioid | bulb


def cardioid_boundary(theta: torch.Tensor) -> torch.Tensor:
    r"""c(θ) = e^{iθ}/2 − e^{2iθ}/4: the edge of the main cardioid, where the fixed point has |multiplier| = 1."""
    w = torch.exp(1j * theta)
    return w / 2 - w * w / 4
