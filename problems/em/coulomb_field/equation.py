r"""Physics → tensor translation.

Coulomb's law plus superposition: the field at a point x is the sum of the fields of all charges q,

    \mathbf E(\mathbf x) = \sum_q k\, Q_q \, \frac{\mathbf x - \mathbf x_q}{\lVert \mathbf x - \mathbf x_q \rVert^3}

In index notation, for grid points g, charges q and spatial components i:

    r_{gqi} = x_{gi} - x_{qi}                       (separation vectors, no sum)
    E_{gi}  = k \, Q_q \, r_{gqi} / |r_{gq}|^3       (q is repeated ⇒ summed: superposition)

with |r_{gq}| = \sqrt{r_{gqj} r_{gqj}} (j repeated ⇒ summed). Broadcasting builds every
separation at once and one einsum adds up all the charges.
"""

import torch


def separations(points: torch.Tensor, charge_pos: torch.Tensor) -> torch.Tensor:
    """r_{gqi} = x_{gi} − x_{qi}.   points: (G, 2), charge_pos: (Q, 2) → (G, Q, 2)"""
    return points[:, None, :] - charge_pos[None, :, :]  # broadcast: (G, 1, 2) − (1, Q, 2)


def electric_field(points, charge_pos, charges, k: float = 1.0) -> torch.Tensor:
    r"""E_{gi} = k Q_q r_{gqi} / |r_{gq}|³   (sum over q).   charges: (Q,) → (G, 2)"""
    r = separations(points, charge_pos)  # (G, Q, 2)
    distance = torch.sqrt(torch.einsum("gqj,gqj->gq", r, r))  # (G, Q)   |r_gq|
    distance = distance.clamp_min(1e-6)  # a grid point exactly on a charge would divide by zero
    return k * torch.einsum("q,gqi,gq->gi", charges, r, 1.0 / distance**3)
