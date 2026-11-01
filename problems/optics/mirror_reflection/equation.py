r"""Physics → tensor translation.

The law of reflection (angle in = angle out) in vector form: for a unit ray direction \mathbf d
hitting a mirror with unit normal \mathbf n,

    \mathbf r = \mathbf d - 2 (\mathbf d\cdot\mathbf n)\, \mathbf n

The dot product picks out the part of d along the normal; subtracting it twice flips that part.
In index notation for a batch of rays r and spatial components i:

    r_{ri} = d_{ri} - 2 (d_{rj} n_{rj})\, n_{ri}        (repeated j ⇒ summed)

For a spherical mirror with centre c_i and radius R the normal at the hit point p_{ri} is

    n_{ri} = (c_i - p_{ri}) / R
"""

import torch


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""a_{rj} b_{rj}: one dot product per ray.   (R, 2), (R, 2) → (R,)"""
    return torch.einsum("rj,rj->r", a, b)


def reflect(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    r"""r_{ri} = d_{ri} - 2 (d_{rj} n_{rj}) n_{ri}.   (R, 2) → (R, 2)"""
    return d - 2 * dot(d, n)[:, None] * n


def sphere_normal(p: torch.Tensor, centre: torch.Tensor, radius: float) -> torch.Tensor:
    r"""n_{ri} = (c_i - p_{ri}) / R: unit normal of a sphere, pointing to its centre.   (R, 2) → (R, 2)"""
    return (centre - p) / radius
