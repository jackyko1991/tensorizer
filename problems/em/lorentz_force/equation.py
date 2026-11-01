r"""Physics → tensor translation.

The Lorentz force on a particle of charge q and mass m,

    m \dot{\mathbf v} = q (\mathbf E + \mathbf v \times \mathbf B),

written in index notation for a batch of particles b and spatial components i, j, k:

    m_b \dot v_{bi} = q_b ( E_i + \varepsilon_{ijk} v_{bj} B_k )

The cross product is the Levi-Civita tensor contracted with two vectors: j and k are
repeated, so both are summed — exactly einsum('ijk,bj,k->bi'). As a first-order system
with state y = (x, v):

    \dot x_{bi} = v_{bi},    \dot v_{bi} = (q_b / m_b)(E_i + \varepsilon_{ijk} v_{bj} B_k)
"""

import torch


def cross(eps: torch.Tensor, v: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""(v × B)_{bi} = \varepsilon_{ijk} v_{bj} B_k.   eps: (3, 3, 3), v: (P, 3), B: (3,) → (P, 3)"""
    return torch.einsum("ijk,bj,k->bi", eps, v, B)


def acceleration(v, q, m, E, B, eps) -> torch.Tensor:
    r"""\dot v_{bi} = (q_b/m_b)(E_i + \varepsilon_{ijk} v_{bj} B_k).   v: (P, 3), q, m: (P,) → (P, 3)"""
    force_per_charge = E + cross(eps, v, B)  # (P, 3)  E_i broadcast over b
    return torch.einsum("b,bi->bi", q / m, force_per_charge)


def rhs(y: torch.Tensor, q, m, E, B, eps) -> torch.Tensor:
    """dy/dt for the state y of shape (P, 2, 3): y[:, 0] = position, y[:, 1] = velocity."""
    x_dot = y[:, 1]
    v_dot = acceleration(y[:, 1], q, m, E, B, eps)
    return torch.stack([x_dot, v_dot], dim=1)


def cyclotron_frequency(q: torch.Tensor, m: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""ω_b = |q_b| \sqrt{B_k B_k} / m_b.   → (P,)"""
    return q.abs() * torch.sqrt(torch.einsum("k,k->", B, B)) / m


def drift_velocity(E: torch.Tensor, B: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""u_i = \varepsilon_{ijk} E_j B_k / (B_l B_l): the E×B drift of every guiding centre.   → (3,)"""
    return torch.einsum("ijk,j,k->i", eps, E, B) / torch.einsum("l,l->", B, B)
