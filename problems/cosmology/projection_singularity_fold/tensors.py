"""Concrete tensors: sample points on the Flamm slice and in Schwarzschild and Kruskal coordinates, plot grids.

All points are batched (B, n) in float64; r is in units of the Schwarzschild radius r_s.
"""

import math

import torch


def flamm_points(rs: float, device) -> torch.Tensor:
    """(B, 2) = (r, phi) on the equatorial slice outside the horizon."""
    r = rs * torch.tensor([1.05, 1.3, 2.0, 3.5, 6.0], dtype=torch.float64)
    phi = torch.tensor([0.0, 1.1, 2.9, 4.4], dtype=torch.float64)
    R, P = torch.meshgrid(r, phi, indexing="ij")
    return torch.stack([R.flatten(), P.flatten()], 1).to(device)


def spacetime_points(rs: float, region: str, device) -> torch.Tensor:
    """(B, 4) = (t, r, theta, phi) outside (1.2 to 8 r_s) or inside (0.1 to 0.9 r_s) the horizon, off the poles."""
    r = rs * (torch.tensor([1.2, 1.7, 3.0, 8.0]) if region == "exterior" else torch.tensor([0.1, 0.35, 0.6, 0.9]))
    t = rs * torch.tensor([-1.3, 0.0, 2.1], dtype=torch.float64)
    th = torch.tensor([0.4, 1.3, 2.6], dtype=torch.float64)
    grid = torch.meshgrid(t, r.double(), th.double(), indexing="ij")
    x = torch.stack([g.flatten().double() for g in grid], 1)  # (B, 3) = (t, r, theta)
    phi = torch.linspace(0.2, 5.9, x.shape[0], dtype=torch.float64)[:, None]
    return torch.cat([x, phi], 1).to(device)


def kruskal_from_schwarzschild(r: torch.Tensor, t: torch.Tensor, rs: float):
    """(T, X) of the region-I (r > r_s) or region-II (r < r_s, future interior) point (t, r)."""
    y = r / rs
    amp = torch.sqrt((y - 1).abs()) * torch.exp(y / 2)
    tau = t / (2 * rs)
    out = y > 1
    T = amp * torch.where(out, torch.sinh(tau), torch.cosh(tau))
    X = amp * torch.where(out, torch.cosh(tau), torch.sinh(tau))
    return T, X


def kruskal_points(rs: float, device) -> torch.Tensor:
    """(B, 4) = (T, X, theta, phi) from 0.2 r_s to 4 r_s, including points on the horizon T = X."""
    r = rs * torch.tensor([0.2, 0.5, 0.9, 0.99, 1.01, 1.5, 4.0], dtype=torch.float64)
    t = rs * torch.tensor([-1.4, 0.0, 1.0], dtype=torch.float64)
    R, Tt = torch.meshgrid(r, t, indexing="ij")
    T, X = kruskal_from_schwarzschild(R.flatten(), Tt.flatten(), rs)
    T = torch.cat([T, torch.tensor([0.3, 1.2, 0.0])])  # on the horizon r = r_s and at the bifurcation point
    X = torch.cat([X, torch.tensor([0.3, 1.2, 0.0])])
    th = torch.linspace(0.5, 2.5, len(T), dtype=torch.float64)
    phi = torch.linspace(0.1, 6.0, len(T), dtype=torch.float64)
    return torch.stack([T.double(), X.double(), th, phi], 1).to(device)


def radius_line(rs: float, n: int, device, r_min: float = 0.02, r_max: float = 6.0) -> torch.Tensor:
    """(n,) logarithmic radii for curvature and tidal curves."""
    return rs * torch.logspace(math.log10(r_min), math.log10(r_max), n, dtype=torch.float64, device=device)
