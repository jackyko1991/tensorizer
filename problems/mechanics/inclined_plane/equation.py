r"""Physics → tensor translation.

A block of mass m on a ramp feels three forces: gravity, the normal force N from the ramp and friction f.
Newton's second law, divided by m:

    \mathbf a = \mathbf g + \mathbf N / m + \mathbf f / m

For a batch of ramps b with angles θ_b we use two unit vectors per ramp (components i ∈ {x, z}):

    s_{bi} = (\cos\theta_b, -\sin\theta_b)     points down the slope
    n_{bi} = (\sin\theta_b,  \cos\theta_b)     points out of the surface

Projecting a vector onto a unit vector is a repeated (summed) index, einsum('i,bi->b'):

    g_{\parallel b} = g_i s_{bi} = g \sin\theta_b,      g_{\perp b} = g_i n_{bi} = -g \cos\theta_b

The ramp pushes back just enough to cancel the normal part of gravity, and friction acts along the slope:

    N_{bi}/m = -g_{\perp b} \, n_{bi}
    f_{bi}/m = -\min(g_{\parallel b}, \mu_s N_b/m) \, s_{bi}       if the block can stay at rest (static)
    f_{bi}/m = -\mu_k (N_b/m) \, s_{bi}                             if it slides (kinetic)

    a_{bi} = g_i + N_{bi}/m + f_{bi}/m

The block stays put when g_∥ ≤ μ_s N/m, i.e. tan θ ≤ μ_s. Otherwise a_∥ = g (sin θ − μ_k cos θ).
"""

import torch


def project(vector: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    r"""Component along each ramp's unit vector:  v_i u_{bi}  (i summed).   (2,), (B, 2) → (B,)"""
    return torch.einsum("i,bi->b", vector, unit)


def normal_force(g: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    r"""N_{bi}/m = -(g_j n_{bj}) n_{bi}: cancels the part of gravity pushing into the ramp.   → (B, 2)"""
    g_perp = project(g, n)  # (B,)  = -g cos θ
    return -torch.einsum("b,bi->bi", g_perp, n)


def friction(g: torch.Tensor, s: torch.Tensor, n: torch.Tensor, mu_s: float, mu_k: float) -> torch.Tensor:
    r"""f_{bi}/m along -s_{bi}: static if it can hold the block, kinetic otherwise.   → (B, 2)"""
    g_par = project(g, s)  # (B,)  pull down the slope, g sin θ
    normal = -project(g, n)  # (B,)  size of N/m, g cos θ
    stuck = g_par <= mu_s * normal  # (B,) True where static friction wins
    size = torch.where(stuck, g_par, mu_k * normal)  # (B,)
    return -torch.einsum("b,bi->bi", size, s)


def acceleration(g: torch.Tensor, s: torch.Tensor, n: torch.Tensor, mu_s: float, mu_k: float) -> torch.Tensor:
    r"""a_{bi} = g_i + N_{bi}/m + f_{bi}/m.   → (B, 2)"""
    return g + normal_force(g, n) + friction(g, s, n, mu_s, mu_k)
