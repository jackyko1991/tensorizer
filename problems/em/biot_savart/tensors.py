"""Build the concrete tensors: Levi-Civita symbol, wire segments and field points in the x–z plane."""

import itertools
import math

import torch

EVEN = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]  # cyclic permutations of (x, y, z)
EXTENT = 1.5  # the plot shows −EXTENT ≤ x, z ≤ EXTENT (metres)

# coaxial loops for each layout as (height in units of R, current direction)
LAYOUTS = {
    "loop": [(0.0, +1)],
    "helmholtz": [(-0.5, +1), (0.5, +1)],  # spacing R: the most uniform field between the coils
    "anti_helmholtz": [(-0.5, +1), (0.5, -1)],  # opposite currents: B = 0 in the middle
}


def levi_civita(device: torch.device) -> torch.Tensor:
    """ε_ijk of shape (3, 3, 3): +1 for even permutations, −1 for odd ones, 0 when an index repeats."""
    eps = torch.zeros(3, 3, 3, device=device)
    for i, j, k in itertools.permutations(range(3)):
        eps[i, j, k] = 1.0 if (i, j, k) in EVEN else -1.0
    return eps


def circle(R: float, z0: float, n: int, direction: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """A loop of radius R in the plane z = z0 cut into n segments: (midpoints (n, 3), dl (n, 3)).

    direction = +1 runs anticlockwise seen from +z (right-hand rule: B points up the axis).
    """
    phi = torch.linspace(0.0, 2 * math.pi, n + 1, device=device)  # n + 1 corners, the last equals the first
    corners = torch.stack([R * torch.cos(phi), R * torch.sin(phi), torch.full_like(phi, z0)], dim=-1)
    if direction < 0:
        corners = corners.flip(0)
    mid = 0.5 * (corners[1:] + corners[:-1])  # (n, 3)
    dl = corners[1:] - corners[:-1]  # (n, 3) segment vectors along the current
    return mid, dl


def coils(layout: str, R: float, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """All loops of a layout joined into one list of segments: (midpoints (S, 3), dl (S, 3))."""
    pieces = [circle(R, height * R, n, sign, device) for height, sign in LAYOUTS[layout]]
    return torch.cat([m for m, _ in pieces]), torch.cat([d for _, d in pieces])


def straight_wire(length: float, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """A straight wire along z from −length/2 to +length/2 in n segments: (midpoints (n, 3), dl (n, 3))."""
    z = torch.linspace(-length / 2, length / 2, n + 1, device=device)
    corners = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)
    return 0.5 * (corners[1:] + corners[:-1]), corners[1:] - corners[:-1]


def plane_xz(n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (coordinates (N,), field points (N·N, 3)) on the plane y = 0; rows = z, columns = x."""
    xs = torch.linspace(-EXTENT, EXTENT, n, device=device)
    Z, X = torch.meshgrid(xs, xs, indexing="ij")
    points = torch.stack([X.reshape(-1), torch.zeros(n * n, device=device), Z.reshape(-1)], dim=-1)
    return xs, points


def z_axis(n: int, device) -> torch.Tensor:
    """n field points on the z axis, shape (n, 3)."""
    z = torch.linspace(-EXTENT, EXTENT, n, device=device)
    return torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1)


def radii(R_mean: float, n_steps: int) -> list[float]:
    """The loop radius at each animation step: it swings between 0.5 R_mean and 1.5 R_mean."""
    return [R_mean * (1 + 0.5 * math.sin(2 * math.pi * k / n_steps)) for k in range(n_steps)]
