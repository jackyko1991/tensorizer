r"""Physics → tensor translation.

A cloth as N point masses (nodes a) joined by E springs (edges e), hanging under gravity with air damping γ:

    m \ddot{\mathbf x}_a = \sum_{\text{springs } e \ni a} \mathbf F_e + m \mathbf g - m \gamma \dot{\mathbf x}_a .

The incidence matrix D_{ea} encodes the mesh: spring e from node i to node j has D_{ei} = −1, D_{ej} = +1.
With it, everything is an index contraction (repeated indices summed; i ∈ {x, y, z}):

    d_{ei} = D_{ea} x_{ai}                         edge vector of every spring, sum over nodes a
    \ell_e = \sqrt{d_{ei} d_{ei}},   u_{ei} = d_{ei} / \ell_e
    F_{ai} = -D_{ea} \, k (\ell_e - \ell^0_e) \, u_{ei}     Hooke's law, sum over springs e

    \ddot x_{ai} = F_{ai} / m + g_i - \gamma \dot x_{ai}        (zero for pinned nodes)

Big meshes store D sparsely; the same sum over e is then a scatter: index_add_ over the edge list.
The energy  E = ½ m \dot x_{ai}\dot x_{ai} + ½ k (\ell_e - \ell^0_e)^2 - m g_i x_{ai}  only decreases when γ > 0
(dE/dt = −m γ \dot x_{ai}\dot x_{ai}) and is conserved when γ = 0.
"""

import torch


def edge_vectors(x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    r"""d_{ei} = D_{ea} x_{ai}.   x: (N, 3), D: (E, N) → (E, 3)"""
    return torch.einsum("ea,ai->ei", D, x)


def spring_forces(x: torch.Tensor, D: torch.Tensor, rest: torch.Tensor, k: float) -> torch.Tensor:
    r"""F_{ai} = -D_{ea} k (\ell_e - \ell^0_e) u_{ei}.   → (N, 3)"""
    d = edge_vectors(x, D)  # (E, 3)
    length = torch.sqrt(torch.einsum("ei,ei->e", d, d))  # (E,)
    tension = k * (length - rest)  # (E,) positive when stretched
    u = d / length[:, None]  # (E, 3) unit vectors i → j
    return -torch.einsum("ea,e,ei->ai", D, tension, u)


def spring_forces_scatter(x: torch.Tensor, edges: torch.Tensor, rest: torch.Tensor, k: float) -> torch.Tensor:
    """Same forces from the edge list (E, 2) with gather + index_add_ — how large meshes do it."""
    i, j = edges[:, 0], edges[:, 1]
    d = x[j] - x[i]  # (E, 3)
    length = d.norm(dim=-1)  # (E,)
    f = (k * (length - rest) / length)[:, None] * d  # (E, 3) pull on node i towards j
    F = torch.zeros_like(x)
    F.index_add_(0, i, f)  # each spring pulls its first node towards the second …
    F.index_add_(0, j, -f)  # … and the second node back (Newton's third law)
    return F


def rhs(y: torch.Tensor, D: torch.Tensor, rest: torch.Tensor, free: torch.Tensor, cloth: dict) -> torch.Tensor:
    """dy/dt for y of shape (2, N, 3): y[0] = positions, y[1] = velocities. free: (N,) 1 = moves, 0 = pinned."""
    x, v = y[0], y[1]
    a = spring_forces(x, D, rest, cloth["k"]) / cloth["m"] + cloth["g"] - cloth["gamma"] * v  # (N, 3)
    a = free[:, None] * a  # pinned nodes never accelerate
    return torch.stack([v, a])


def energy(y: torch.Tensor, D: torch.Tensor, rest: torch.Tensor, cloth: dict) -> torch.Tensor:
    r"""E = ½ m \dot x_{ai}\dot x_{ai} + ½ k (\ell_e − \ell^0_e)^2 − m g_i x_{ai}.   y: (T, 2, N, 3) → (T,)"""
    x, v = y[:, 0], y[:, 1]  # (T, N, 3)
    kinetic = 0.5 * cloth["m"] * torch.einsum("tai,tai->t", v, v)
    d = torch.einsum("ea,tai->tei", D, x)  # (T, E, 3)
    stretch = d.norm(dim=-1) - rest  # (T, E)
    elastic = 0.5 * cloth["k"] * (stretch**2).sum(-1)
    gravity = -cloth["m"] * torch.einsum("i,tai->t", cloth["g"], x)
    return kinetic + elastic + gravity
