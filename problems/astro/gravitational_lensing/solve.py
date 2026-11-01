"""Trace the null geodesics with RK4 in φ, and lens the moving source on the whole pixel grid at once."""

import math

import torch

from problems._common.integrators import integrate

from .equation import binet_rhs, lens_map, source_brightness


def trace_rays(y0: torch.Tensor, M: float, dphi: float = 0.004, phi_max: float = 4 * math.pi) -> dict:
    """Integrate u(φ) for every ray. A ray is done when it escapes (u < 0) or falls in (u > 1/(2M): horizon)."""
    n = int(round(phi_max / dphi))
    traj = integrate(lambda y, phi: binet_rhs(y, M), y0, dphi, n)  # (T, B, 2)
    u = traj[:, :, 0]
    phi = torch.arange(n + 1, dtype=u.dtype, device=u.device) * dphi
    escaped = (u < 0) & (phi[:, None] > 0.5)
    captured = u > 1 / (2 * M)
    done = torch.cummax((escaped | captured).int(), dim=0).values.bool()
    return {"phi": phi, "u": u.masked_fill(done, float("nan")), "escaped": escaped.any(0), "raw_u": u}


def deflection_angles(rays: dict) -> torch.Tensor:
    """α = φ_out − π, with φ_out where u returns to zero (linear interpolation); NaN for captured rays, (B,)."""
    u, phi = rays["raw_u"], rays["phi"]
    alpha = torch.full((u.shape[1],), float("nan"), dtype=u.dtype)
    for b in range(u.shape[1]):
        crossing = torch.nonzero((u[:-1, b] > 0) & (u[1:, b] <= 0) & (phi[1:] > 0.5)).flatten()
        if len(crossing) and rays["escaped"][b]:
            k = int(crossing[0])
            f = u[k, b] / (u[k, b] - u[k + 1, b])
            alpha[b] = phi[k] + f * (phi[k + 1] - phi[k]) - math.pi
    return alpha


def ray_paths(rays: dict) -> torch.Tensor:
    """Cartesian points (T, B, 2) of each ray, coming in from the left: x = −cos φ / u, y = sin φ / u."""
    r = 1 / rays["u"]
    phi = rays["phi"][:, None]
    return torch.stack([-r * torch.cos(phi), r * torch.sin(phi)], dim=-1)


def lensed_images(theta: torch.Tensor, centres: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Lensed and unlensed images (F, n, n) for each source position (F, 2), all frames in one broadcast."""
    beta = lens_map(theta)  # (n, n, 2): where each pixel looks in the source plane
    lensed = source_brightness(beta[None], centres[:, None, None, :], radius)  # (F, n, n)
    plain = source_brightness(theta[None], centres[:, None, None, :], radius)  # (F, n, n) without the lens
    return lensed, plain


def magnification(lensed: torch.Tensor, plain: torch.Tensor) -> torch.Tensor:
    """Total flux ratio per frame (pixel areas cancel), (F,)."""
    return lensed.sum(dim=(-2, -1)) / plain.sum(dim=(-2, -1))
