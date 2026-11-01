"""Build the concrete tensors: gradient directions, a 2D slice of voxels with fibre bundles, the true D_ij field."""

import math

import torch

# Approximate diffusivities in 10⁻³ mm²/s (= µm²/ms). For teaching only.
LAMBDA_PARALLEL = 1.7  # along an axon bundle
LAMBDA_PERP = 0.3  # across it
D_CSF = 3.0  # free water at body temperature
D_GREY = 0.8  # isotropic tissue between the bundles


def gradient_directions(n: int, device: torch.device) -> torch.Tensor:
    """ĝ_{gi}: n unit vectors spread evenly over a hemisphere (Fibonacci spiral). Shape (G, 3).

    ±ĝ give the same signal, so one hemisphere is enough.
    """
    k = torch.arange(n, dtype=torch.float64, device=device) + 0.5
    z = 1 - k / n  # from ~1 down to ~0: upper hemisphere
    phi = math.pi * (3 - math.sqrt(5)) * k  # golden-angle steps in azimuth
    r = torch.sqrt(1 - z**2)
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def fibre_tensor(e: torch.Tensor, lam_par: float, lam_perp: float) -> torch.Tensor:
    """D_{ij} = λ∥ e_i e_j + λ⊥ (δ_ij − e_i e_j) for unit fibre directions e (..., 3) → (..., 3, 3)."""
    eye = torch.eye(3, dtype=e.dtype, device=e.device)
    ee = torch.einsum("...i,...j->...ij", e, e)
    return lam_par * ee + lam_perp * (eye - ee)


def phantom(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A slice of n × n voxels: an arched bundle, a vertical bundle crossing it, and a CSF-filled ventricle.

    Returns the true tensors D_{vij} (V, 3, 3) with V = n², the fibre direction of each voxel (V, 3; zero where
    there is no single fibre) and a label per voxel (V,): 0 grey matter, 1 arc, 2 vertical, 3 crossing, 4 CSF.
    Voxel v = row · n + column; the coordinates (x, y) = (column, row) with y pointing up.
    """
    c = torch.arange(n, dtype=torch.float64, device=device)
    y, x = torch.meshgrid(c, c, indexing="ij")
    s = n / 20  # sizes below are for a 20-voxel slice
    cx, cy = 9.5 * s, 0.0
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    arc = (r > 10 * s) & (r < 14 * s)
    vertical = (x > 13.5 * s) & (x < 16.5 * s)
    csf = ((x - cx) / (3.0 * s)) ** 2 + ((y - 4 * s) / (2.2 * s)) ** 2 < 1

    zero = torch.zeros_like(x)
    e_arc = torch.stack([-(y - cy), x - cx, zero], dim=-1) / r.clamp(min=1e-9)[..., None]  # tangent to circles
    e_up = torch.stack([zero, torch.ones_like(x), zero], dim=-1)

    D = D_GREY * torch.eye(3, dtype=torch.float64, device=device).expand(n, n, 3, 3).clone()
    D_arc = fibre_tensor(e_arc, LAMBDA_PARALLEL, LAMBDA_PERP)
    D_up = fibre_tensor(e_up, LAMBDA_PARALLEL, LAMBDA_PERP)
    D = torch.where(arc[..., None, None], D_arc, D)
    D = torch.where(vertical[..., None, None], D_up, D)
    D = torch.where((arc & vertical)[..., None, None], 0.5 * (D_arc + D_up), D)  # crossing: average of the two
    D = torch.where(csf[..., None, None], D_CSF * torch.eye(3, dtype=torch.float64, device=device), D)

    label = torch.zeros(n, n, dtype=torch.long, device=device)
    label[arc], label[vertical], label[arc & vertical], label[csf] = 1, 2, 3, 4
    fibre = torch.where((label == 1)[..., None], e_arc, torch.zeros_like(e_arc))
    fibre = torch.where((label == 2)[..., None], e_up, fibre)
    return D.reshape(-1, 3, 3), fibre.reshape(-1, 3), label.reshape(-1)


def design_matrix(g: torch.Tensor) -> torch.Tensor:
    """B_{gm}: ĝ_i D_ij ĝ_j = B_{gm} d_m for the 6 unique components d = (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz). (G, 6)"""
    gx, gy, gz = g.unbind(-1)
    return torch.stack([gx * gx, gy * gy, gz * gz, 2 * gx * gy, 2 * gx * gz, 2 * gy * gz], dim=-1)


def seeds(n: int, device: torch.device) -> torch.Tensor:
    """Tractography start points (P, 2) in voxel coordinates: across the arc's top and the vertical bundle's foot."""
    s = n / 20
    across_arc = torch.stack([torch.full((6,), 9.5 * s), torch.linspace(10.4 * s, 13.6 * s, 6)], dim=-1)
    foot = torch.stack([torch.linspace(13.9 * s, 16.1 * s, 4), torch.full((4,), 1.0 * s)], dim=-1)
    return torch.cat([across_arc, foot]).to(device=device, dtype=torch.float64)
