r"""Physics → tensor translation.

The Biot–Savart law gives the magnetic field of a steady current I flowing along a wire,

    \mathbf B(\mathbf x) = \frac{\mu_0 I}{4\pi} \oint \frac{d\mathbf l \times \mathbf r}{\lVert \mathbf r \rVert^3},
    \qquad \mathbf r = \mathbf x - \mathbf x_{\text{wire}}

Cut the wire into S short straight segments s with midpoint x_{si} and direction vector dl_{si}.
For field points g the integral becomes a sum over s, and the cross product a Levi-Civita contraction:

    r_{gsk} = x_{gk} - x_{sk}
    B_{gi}  = \frac{\mu_0 I}{4\pi} \, \varepsilon_{ijk} \, dl_{sj} \, r_{gsk} \, / \, |r_{gs}|^3

j, k and s are repeated, so all three are summed — one einsum('ijk,sj,gsk,gs->gi') does the
cross product and adds up every segment of the wire.

Check: on the axis of a circular loop of radius R the field points along the axis with
B_z = μ0 I R² / (2 (R² + z²)^{3/2}).
"""

import math

import torch

MU0 = 4e-7 * math.pi  # T·m/A


def biot_savart(points, mid, dl, eps, current: float, softening: float = 0.0) -> torch.Tensor:
    r"""B_{gi} = (μ0 I/4π) ε_{ijk} dl_{sj} r_{gsk} / |r_{gs}|³.

    points: (G, 3), mid, dl: (S, 3), eps: (3, 3, 3) → (G, 3)
    softening (m) keeps the field finite for grid points that land right on the wire.
    """
    r = points[:, None, :] - mid[None, :, :]  # (G, S, 3) from each segment to each point
    r2 = torch.einsum("gsk,gsk->gs", r, r) + softening**2  # (G, S)   |r|²
    inv_r3 = r2**-1.5  # (G, S)
    return MU0 * current / (4 * math.pi) * torch.einsum("ijk,sj,gsk,gs->gi", eps, dl, r, inv_r3)


def loop_axis_field(z: torch.Tensor, R: float, z_loops: list[float], currents: list[float]) -> torch.Tensor:
    """Analytic B_z on the common axis of coaxial loops of radius R: Σ_l μ0 I_l R² / (2 (R² + (z − z_l)²)^{3/2})."""
    total = torch.zeros_like(z)
    for z_l, I_l in zip(z_loops, currents):
        total = total + MU0 * I_l * R**2 / (2 * (R**2 + (z - z_l) ** 2) ** 1.5)
    return total


def wire_field(r: torch.Tensor, current: float) -> torch.Tensor:
    """Analytic |B| = μ0 I / (2π r) at distance r from an infinitely long straight wire."""
    return MU0 * current / (2 * math.pi * r)
