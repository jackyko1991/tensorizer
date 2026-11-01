"""Build the concrete tensors: the grid layout, the sponge damping map and the barrier with two slits.

Everything is measured in grid cells (h = 1) and the wave speed is c = 1. Arrays are indexed
u[y, x] so they can go straight into imshow.
"""

from dataclasses import dataclass

import torch

NY, NX = 240, 260  # grid size in cells
SPONGE = 20  # width of the absorbing frame
SOURCE_X = 25  # column of the plane-wave line source
BARRIER_X = 60  # first column of the barrier
BARRIER_THICKNESS = 3


@dataclass
class Setup:
    sigma: torch.Tensor  # (Ny, Nx) damping rate σ_ab, zero in the interior
    wall: torch.Tensor  # (Ny, Nx) bool, True where the barrier is
    screen_x: int  # column where the intensity is recorded


def sponge(device: torch.device, strength: float = 0.5) -> torch.Tensor:
    """σ_{ab} grows quadratically from 0 at the inner edge of the frame to `strength` at the border."""
    y = torch.arange(NY, device=device, dtype=torch.float32)
    x = torch.arange(NX, device=device, dtype=torch.float32)
    depth_y = torch.clamp(SPONGE - torch.minimum(y, NY - 1 - y), min=0) / SPONGE  # (Ny,) 0 inside → 1 at edge
    depth_x = torch.clamp(SPONGE - torch.minimum(x, NX - 1 - x), min=0) / SPONGE  # (Nx,)
    depth = torch.maximum(depth_y[:, None], depth_x[None, :])  # (Ny, Nx)
    return strength * depth**2


def barrier(slit_sep: float, slit_width: float, device: torch.device) -> torch.Tensor:
    """Wall mask (Ny, Nx): a thick vertical wall with two openings centred at y = NY/2 ± slit_sep/2."""
    y = torch.arange(NY, device=device, dtype=torch.float32)[:, None]  # (Ny, 1)
    x = torch.arange(NX, device=device)[None, :]  # (1, Nx)
    centre = NY / 2
    in_slit = ((y - centre - slit_sep / 2).abs() < slit_width / 2) | (
        (y - centre + slit_sep / 2).abs() < slit_width / 2
    )
    in_wall_columns = (x >= BARRIER_X) & (x < BARRIER_X + BARRIER_THICKNESS)
    return in_wall_columns & ~in_slit  # (Ny, Nx) by broadcasting


def setup(slit_sep: float, slit_width: float, screen_dist: float, device: torch.device) -> Setup:
    screen_x = min(BARRIER_X + int(round(screen_dist)), NX - SPONGE - 1)
    return Setup(sponge(device), barrier(slit_sep, slit_width, device), screen_x)
