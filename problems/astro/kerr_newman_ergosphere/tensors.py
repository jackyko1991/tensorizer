"""Build the concrete tensors: a batched (M, a, Q) grid, the sweep path through extremality and angle grids.

Units G = c = 1: lengths in GM/c², spin a = J/(Mc), charge Q in geometrised units (Q² ↔ G Q²/(4πε₀ c⁴)).
"""

import math

import torch

F64 = torch.float64


def parameter_grid(n: int = 61, masses=(1.0, 2.0, 5.0), top: float = 1.3, device=None):
    """Every combination of mass, spin and charge as one batch: M, a, Q of shape (n_M, n, n).

    a/M and Q/M run from 0 to `top`, so the grid crosses the extremal circle a² + Q² = M².
    """
    m = torch.tensor(masses, dtype=F64, device=device)  # (n_M,)
    f = torch.linspace(0, top, n, dtype=F64, device=device)  # (n,) fractions of M
    M = m[:, None, None].expand(-1, n, n)  # (n_M, n, n)
    a = M * f[None, :, None]  # spin varies along axis 1
    Q = M * f[None, None, :]  # charge along axis 2
    return M, a, Q


def sweep(angle_deg: float, chi_max: float, n: int, device=None):
    """Path from Schwarzschild through extremality: χ = √(a² + Q²)/M from 0 to χ_max at a fixed mixing angle
    ψ (0° = pure spin, Kerr; 90° = pure charge, Reissner–Nordström). Returns χ, M, a, Q, each (n,)."""
    psi = math.radians(angle_deg)
    n_hole = max(2, round(n / chi_max))  # frames up to chi = 1, which is hit exactly (the extremal hole)
    chi = torch.cat([torch.linspace(0, 1, n_hole, dtype=F64, device=device),
                     torch.linspace(1, chi_max, n - n_hole + 1, dtype=F64, device=device)[1:]])  # fmt: skip
    M = torch.ones_like(chi)
    return chi, M, chi * math.cos(psi) * M, chi * math.sin(psi) * M


def polar_angles(n: int = 241, device=None) -> torch.Tensor:
    """θ from the north pole (0) to the south pole (π): (n,)."""
    return torch.linspace(0, math.pi, n, dtype=F64, device=device)
