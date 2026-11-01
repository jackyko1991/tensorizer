r"""Physics → tensor translation.

Newton's law of gravitation for N bodies (with a small softening length ε),

    \ddot{\mathbf r}_a = G \sum_{b \neq a} m_b
        \frac{\mathbf r_b - \mathbf r_a}{(\lVert \mathbf r_b - \mathbf r_a \rVert^2 + \varepsilon^2)^{3/2}},

written in index notation for bodies a, b and spatial components i, j:

    r_{abi} = x_{bi} - x_{ai}                                   (separation tensor, all pairs)
    \ddot x_{ai} = G \, m_b \, (r_{abj} r_{abj} + \varepsilon^2)^{-3/2} \, r_{abi}

Repeated indices are summed: j gives the squared distance of each pair, einsum('abj,abj->ab'),
and b adds up the pull of every other body on body a, einsum('b,ab,abi->ai').
The term b = a vanishes because r_{aai} = 0, so we simply zero the diagonal.

Conserved quantities: total momentum P_i = m_a v_{ai} and the energy

    E = \tfrac12 m_a v_{ai} v_{ai} - \tfrac12 G \sum_{a \neq b} m_a m_b (r_{abj} r_{abj} + \varepsilon^2)^{-1/2}
"""

import torch


def separations(x: torch.Tensor) -> torch.Tensor:
    r"""r_{abi} = x_{bi} - x_{ai}.   x: (N, 3) → (N, N, 3) by broadcasting."""
    return x[None, :, :] - x[:, None, :]


def acceleration(x: torch.Tensor, m: torch.Tensor, G: float, eps: float) -> torch.Tensor:
    r"""\ddot x_{ai} = G m_b (r_{abj} r_{abj} + ε^2)^{-3/2} r_{abi}.   x: (N, 3), m: (N,) → (N, 3)"""
    r = separations(x)  # (N, N, 3)
    dist_sq = torch.einsum("abj,abj->ab", r, r) + eps**2  # (N, N)
    inv_d3 = dist_sq ** (-1.5)  # (N, N)
    inv_d3.fill_diagonal_(0.0)  # no self-force (also avoids 1/0 when ε = 0)
    return G * torch.einsum("b,ab,abi->ai", m, inv_d3, r)  # sum over b


def energy(x: torch.Tensor, v: torch.Tensor, m: torch.Tensor, G: float, eps: float) -> torch.Tensor:
    r"""E = ½ m_a v_{ai} v_{ai} - ½ G Σ_{a≠b} m_a m_b / \sqrt{r_{abj} r_{abj} + ε^2}.   x, v: (N, 3) → scalar"""
    kinetic = 0.5 * torch.einsum("a,ai,ai->", m, v, v)
    r = separations(x)
    inv_d = (torch.einsum("abj,abj->ab", r, r) + eps**2) ** (-0.5)  # (N, N)
    inv_d.fill_diagonal_(0.0)
    potential = -0.5 * G * torch.einsum("a,b,ab->", m, m, inv_d)  # ½ because each pair appears twice
    return kinetic + potential


def momentum(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    r"""P_i = m_a v_{ai} (sum over bodies a).   v: (..., N, 3) → (..., 3)"""
    return torch.einsum("a,...ai->...i", m, v)
