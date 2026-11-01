"""Build the concrete tensors: gravity, the block densities, their initial state and a grid of tank points."""

import torch


def gravity(g: float, device: torch.device) -> torch.Tensor:
    """g_i with i ∈ {x, z}: shape (2,)."""
    return torch.tensor([0.0, -g], device=device)


def densities(n: int, rho_min: float, rho_max: float, device: torch.device) -> torch.Tensor:
    """n block densities spread evenly between rho_min and rho_max (kg/m³): shape (B,)."""
    return torch.linspace(rho_min, rho_max, n, device=device)


def initial_state(n: int, z0: float, device: torch.device) -> torch.Tensor:
    """y0 of shape (B, 2): every block starts at rest with its bottom face at height z0 above the water."""
    y0 = torch.zeros(n, 2, device=device)
    y0[:, 0] = z0
    return y0


def tank_points(width: float, depth: float, nx: int, nz: int, device: torch.device) -> torch.Tensor:
    """Points x_i filling the water, shape (nz, nx, 2): x from 0 to width, z from -depth up to the surface 0."""
    x = torch.linspace(0.0, width, nx, device=device)
    z = torch.linspace(-depth, 0.0, nz, device=device)
    zz, xx = torch.meshgrid(z, x, indexing="ij")  # (nz, nx) each
    return torch.stack([xx, zz], dim=-1)
