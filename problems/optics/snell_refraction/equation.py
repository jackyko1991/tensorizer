r"""Physics → tensor translation.

Snell's law  n_1 \sin\theta_1 = n_2 \sin\theta_2  in vector form, for a unit direction
\mathbf d hitting a surface with unit normal \mathbf n (pointing back towards the incoming ray):

    \mathbf t = \eta\,\mathbf d + (\eta\cos\theta_i - \cos\theta_t)\,\mathbf n,
    \eta = n_1/n_2,  \cos\theta_i = -\mathbf n\cdot\mathbf d,
    \cos^2\theta_t = 1 - \eta^2 (1 - \cos^2\theta_i)

In index notation for a batch of rays r and spatial components i:

    c_r = -n_{rj} d_{rj}                       (repeated j ⇒ summed: a dot product per ray)
    t_{ri} = \eta_r d_{ri} + (\eta_r c_r - \sqrt{1 - \eta_r^2 (1 - c_r^2)})\, n_{ri}

When the square root's argument is negative there is no refracted ray: total internal
reflection, and the ray is mirrored instead,

    r_{ri} = d_{ri} - 2 (d_{rj} n_{rj})\, n_{ri}
"""

import torch


def cos_incidence(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    r"""c_r = -n_{rj} d_{rj}.   d, n: (R, 2) → (R,)"""
    return -torch.einsum("rj,rj->r", n, d)


def reflect(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    r"""r_{ri} = d_{ri} - 2 (d_{rj} n_{rj}) n_{ri}.   (R, 2) → (R, 2)"""
    d_dot_n = torch.einsum("rj,rj->r", d, n)  # (R,)
    return d - 2 * torch.einsum("r,ri->ri", d_dot_n, n)


def refract(d: torch.Tensor, n: torch.Tensor, eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Vector Snell's law for every ray at once.

    d: (R, 2) unit directions, n: (R, 2) unit normals facing the incoming ray, eta: (R,) = n_1/n_2.
    Returns (new directions (R, 2), tir (R,) bool mask of totally internally reflected rays).
    """
    cos_i = cos_incidence(d, n)  # (R,)
    cos_t_sq = 1 - eta**2 * (1 - cos_i**2)  # (R,)   < 0  ⇔  total internal reflection
    tir = cos_t_sq < 0
    cos_t = torch.sqrt(cos_t_sq.clamp(min=0))  # (R,)
    t = torch.einsum("r,ri->ri", eta, d) + torch.einsum("r,ri->ri", eta * cos_i - cos_t, n)  # (R, 2)
    return torch.where(tir[:, None], reflect(d, n), t), tir


def sin_angle(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    r"""|\sin\theta| between direction and normal = |2D cross product| = |d_x n_y - d_y n_x|.   (R, 2) → (R,)"""
    return (d[:, 0] * n[:, 1] - d[:, 1] * n[:, 0]).abs()
