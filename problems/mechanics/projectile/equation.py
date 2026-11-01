r"""Physics → tensor translation.

Newton's second law with quadratic drag,

    m \ddot{\mathbf r} = m \mathbf g - c \lVert \dot{\mathbf r} \rVert \dot{\mathbf r},

written in index notation for a batch of balls b and spatial components i:

    \ddot x_{bi} = g_i - (c/m) \sqrt{v_{bj} v_{bj}} \, v_{bi}

A repeated index (j above) is summed — that is exactly what einsum('bj,bj->b') does.
As a first-order system with state y = (x, v):

    \dot x_{bi} = v_{bi},    \dot v_{bi} = \ddot x_{bi}
"""

import torch


def acceleration(v: torch.Tensor, g: torch.Tensor, k: float) -> torch.Tensor:
    r"""\ddot x_{bi} = g_i - k \sqrt{v_{bj} v_{bj}} v_{bi}   with k = c/m.   v: (B, 2) → (B, 2)"""
    speed = torch.sqrt(torch.einsum("bj,bj->b", v, v))  # (B,)   |v_b|
    return g - k * torch.einsum("b,bi->bi", speed, v)  # (B, 2) broadcast g_i over b


def rhs(y: torch.Tensor, g: torch.Tensor, k: float) -> torch.Tensor:
    """dy/dt for the state y of shape (B, 2, 2): y[:, 0] = position, y[:, 1] = velocity."""
    x_dot = y[:, 1]
    v_dot = acceleration(y[:, 1], g, k)
    return torch.stack([x_dot, v_dot], dim=1)


def energy(y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    r"""Mechanical energy per unit mass  E_b = ½ v_{bi} v_{bi} - g_i x_{bi}  (conserved when k = 0)."""
    x, v = y[..., 0, :], y[..., 1, :]
    return 0.5 * torch.einsum("...i,...i->...", v, v) - torch.einsum("i,...i->...", g, x)
