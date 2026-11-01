r"""Physics → tensor translation.

An ideal gas of N hard disks a = 1..N with positions x_{ai} and velocities v_{ai} (i ∈ {x, y}).
Between collisions every particle moves in a straight line. Two touching disks a, b exchange the
component of their relative velocity along the line joining them (elastic, equal masses):

    d_{abi} = x_{ai} - x_{bi},        n_{abi} = d_{abi} / \sqrt{d_{abj} d_{abj}}

    v'_{ai} = v_{ai} - (\Delta v_{abj} n_{abj}) \, n_{abi},     v'_{bi} = v_{bi} + (\Delta v_{abj} n_{abj}) \, n_{abi}

with \Delta v_{abj} = v_{aj} - v_{bj}. The repeated j is summed: it is the projection of the relative
velocity onto the collision normal, einsum('pj,pj->p'). Both the momentum and the kinetic energy
E = ½ m v_{ai} v_{ai} (summed over a and i) are unchanged by this rule.

Statistics: in 2D each particle carries kT of kinetic energy (½kT per direction), the speeds relax to the
Maxwell–Boltzmann distribution f(v) = (m v / kT) e^{-m v² / 2kT}, and the wall pressure obeys P A = N k T.
"""

import torch


def pair_displacement(x: torch.Tensor) -> torch.Tensor:
    """d_{abi} = x_{ai} − x_{bi} by broadcasting (N, 1, 2) − (1, N, 2).   x: (N, 2) → (N, N, 2)"""
    return x[:, None, :] - x[None, :, :]


def colliding_pairs(x: torch.Tensor, v: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices (a, b) of disks that overlap and are moving towards each other.  Each disk in at most one pair."""
    d = pair_displacement(x)  # (N, N, 2)
    dv = pair_displacement(v)  # (N, N, 2)  Δv_{abi}
    dist2 = torch.einsum("abi,abi->ab", d, d)  # (N, N)  squared distance
    approaching = torch.einsum("abi,abi->ab", dv, d) < 0  # (N, N)  Δv · d < 0
    hit = (dist2 < (2 * radius) ** 2) & approaching
    a, b = torch.nonzero(torch.triu(hit, diagonal=1), as_tuple=True)  # (P,), (P,)  each pair once, a < b

    # A disk touching two others at once waits until the next step, so every update is a clean two-body one.
    count = torch.bincount(torch.cat([a, b]), minlength=len(x))  # (N,)  pairs each disk appears in
    alone = (count[a] == 1) & (count[b] == 1)
    return a[alone], b[alone]


def collide(x: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Elastic collisions of the pairs (a_p, b_p):  v'_{ai} = v_{ai} − (Δv_{pj} n_{pj}) n_{pi}.  → new v (N, 2)"""
    d = x[a] - x[b]  # (P, 2)
    n = d / torch.sqrt(torch.einsum("pj,pj->p", d, d))[:, None]  # (P, 2)  unit collision normal
    w = torch.einsum("pj,pj->p", v[a] - v[b], n)  # (P,)  relative velocity along the normal
    v = v.clone()
    v[a] = v[a] - w[:, None] * n
    v[b] = v[b] + w[:, None] * n
    return v


def bounce_walls(x: torch.Tensor, v: torch.Tensor, box: float, radius: float, m: float = 1.0):
    """Reflect v_i of disks that hit a wall; return (new v (N, 2), momentum given to the walls Σ 2 m |v_i|)."""
    hit = ((x < radius) & (v < 0)) | ((x > box - radius) & (v > 0))  # (N, 2)  wall normal along i
    impulse = 2 * m * v.abs()[hit].sum()
    return torch.where(hit, -v, v), impulse


def kinetic_energy(v: torch.Tensor, m: float = 1.0) -> torch.Tensor:
    """E = ½ m v_{ai} v_{ai}, summed over particles a and components i.  (..., N, 2) → (...)"""
    return 0.5 * m * torch.einsum("...ai,...ai->...", v, v)


def temperature(v: torch.Tensor, m: float = 1.0) -> torch.Tensor:
    """Equipartition in 2D: E = N kT, so kT = E / N  (with k = 1)."""
    return kinetic_energy(v, m) / v.shape[-2]


def maxwell_boltzmann_2d(speed: torch.Tensor, kT: float, m: float = 1.0) -> torch.Tensor:
    """Probability density of the speed in 2D:  f(v) = (m v / kT) exp(−m v² / 2kT)."""
    return m * speed / kT * torch.exp(-m * speed**2 / (2 * kT))
