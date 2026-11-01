r"""Physics → tensor translation.

The Lorenz system (a three-mode model of convection),

    \dot x = \sigma (y - x),\qquad \dot y = x(\rho - z) - y,\qquad \dot z = x y - \beta z,

has a right-hand side that is a linear part plus a quadratic part. For a batch b of trajectories with
state x_{bi} (i = 0, 1, 2 for x, y, z) it reads, in index notation,

    \dot x_{bi} = A_{ij} x_{bj} + Q_{ijk} x_{bj} x_{bk}

with the constant matrix A_{ij} and the rank-3 tensor Q_{ijk} (only two entries are non-zero:
Q_{102} = -1 gives -xz in \dot y, and Q_{201} = +1 gives +xy in \dot z).
Repeated j (and k) are summed: einsum('ij,bj->bi', A, x) and einsum('ijk,bj,bk->bi', Q, x, x).

The divergence of the flow, \partial \dot x_i / \partial x_i = A_{ii} + (Q_{iik} + Q_{iki}) x_k,
is the rate at which a small volume of states grows; for Lorenz it is the constant -(\sigma + 1 + \beta).
"""

import torch


def rhs(x: torch.Tensor, A: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    r"""\dot x_{bi} = A_{ij} x_{bj} + Q_{ijk} x_{bj} x_{bk}.   x: (B, 3), A: (3, 3), Q: (3, 3, 3) → (B, 3)"""
    linear = torch.einsum("ij,bj->bi", A, x)  # sum over j
    quadratic = torch.einsum("ijk,bj,bk->bi", Q, x, x)  # sum over j and k
    return linear + quadratic


def rhs_explicit(x: torch.Tensor, sigma: float, rho: float, beta: float) -> torch.Tensor:
    """The textbook form, component by component (used to check the tensor form).   x: (B, 3) → (B, 3)"""
    X, Y, Z = x.unbind(-1)
    return torch.stack([sigma * (Y - X), X * (rho - Z) - Y, X * Y - beta * Z], dim=-1)


def divergence(x: torch.Tensor, A: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    r"""\partial \dot x_i / \partial x_i = A_{ii} + (Q_{iik} + Q_{iki}) x_{bk}.   x: (B, 3) → (B,)"""
    trace_A = torch.einsum("ii->", A)
    return trace_A + torch.einsum("iik,bk->b", Q, x) + torch.einsum("iki,bk->b", Q, x)


def spread(x: torch.Tensor) -> torch.Tensor:
    r"""RMS distance from the mean, s = \sqrt{\langle (x_{bi} - \bar x_i)(x_{bi} - \bar x_i) \rangle_b}.

    x: (T, B, 3) → (T,)
    """
    d = x - x.mean(dim=1, keepdim=True)  # (T, B, 3)
    return torch.einsum("tbi,tbi->t", d, d).div(x.shape[1]).sqrt()
