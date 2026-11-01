r"""Physics → tensor translation.

Newtonian gravity for three bodies in a plane, for K runs at once (run 0 = reference, run 1 = twin):

    \ddot{\mathbf r}_a = G \sum_{b \neq a} m_b (\mathbf r_b - \mathbf r_a) / \lVert \mathbf r_b - \mathbf r_a \rVert^3

In index notation, with run k, bodies a, b and plane components i, j ∈ {x, y}:

    r_{kabi} = x_{kbi} - x_{kai}                                (separation tensor, all pairs of every run)
    \ddot x_{kai} = G \, m_b \, (r_{kabj} r_{kabj})^{-3/2} \, r_{kabi}

Repeated indices are summed: j gives the squared distance of each pair, einsum('kabj,kabj->kab'),
and b adds up the pull of the other bodies on body a, einsum('b,kab,kabi->kai').
The run index k is never summed: the runs evolve side by side, independently.

Chaos is measured by the phase-space distance between two runs,

    |\Delta| = \sqrt{\Delta x_{ai}\Delta x_{ai} + \Delta v_{ai}\Delta v_{ai}},   \Delta x_{ai} = x_{1ai} - x_{0ai},

which grows like e^{\lambda t} (λ = largest Lyapunov exponent) until it is as large as the orbits themselves.
"""

import torch


def separations(x: torch.Tensor) -> torch.Tensor:
    r"""r_{kabi} = x_{kbi} - x_{kai}.   x: (K, N, 2) → (K, N, N, 2) by broadcasting."""
    return x[:, None, :, :] - x[:, :, None, :]


def pair_distances(x: torch.Tensor) -> torch.Tensor:
    r"""\sqrt{r_{kabj} r_{kabj}}.   x: (K, N, 2) → (K, N, N); the diagonal (a = b) is 0."""
    r = separations(x)
    return torch.einsum("kabj,kabj->kab", r, r).sqrt()


def acceleration(x: torch.Tensor, m: torch.Tensor, G: float) -> torch.Tensor:
    r"""\ddot x_{kai} = G m_b (r_{kabj} r_{kabj})^{-3/2} r_{kabi}.   x: (K, N, 2), m: (N,) → (K, N, 2)"""
    r = separations(x)  # (K, N, N, 2)
    dist_sq = torch.einsum("kabj,kabj->kab", r, r)  # (K, N, N)
    inv_d3 = dist_sq ** (-1.5)  # (K, N, N); the diagonal a = b is 1/0 = inf ...
    inv_d3.diagonal(dim1=1, dim2=2).zero_()  # ... and is set to 0: no self-force
    return G * torch.einsum("b,kab,kabi->kai", m, inv_d3, r)  # sum over b


def rhs(y: torch.Tensor, m: torch.Tensor, G: float) -> torch.Tensor:
    """dy/dt for y = (x, v) stacked: (2, K, N, 2) → (2, K, N, 2)."""
    x, v = y[0], y[1]
    return torch.stack([v, acceleration(x, m, G)])


def energy(x: torch.Tensor, v: torch.Tensor, m: torch.Tensor, G: float) -> torch.Tensor:
    r"""E_k = ½ m_a v_{kai} v_{kai} - ½ G Σ_{a≠b} m_a m_b / |r_{kab}|.   x, v: (..., K, N, 2) → (..., K)"""
    kinetic = 0.5 * torch.einsum("a,...ai,...ai->...", m, v, v)
    r = x[..., None, :, :] - x[..., :, None, :]  # (..., N, N, 2)
    dist = torch.einsum("...abj,...abj->...ab", r, r).sqrt()
    eye = torch.eye(len(m), dtype=torch.bool, device=x.device)
    inv_d = torch.where(eye, 0.0, 1.0 / torch.where(eye, 1.0, dist))
    potential = -0.5 * G * torch.einsum("a,b,...ab->...", m, m, inv_d)  # ½: each pair appears twice
    return kinetic + potential


def momentum(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    r"""P_i = m_a v_{ai} (sum over bodies a).   v: (..., N, 2) → (..., 2)"""
    return torch.einsum("a,...ai->...i", m, v)


def center_of_mass(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    r"""X_i = m_a x_{ai} / M.   x: (..., N, 2) → (..., 2)"""
    return torch.einsum("a,...ai->...i", m, x) / m.sum()


def phase_space_distance(xs: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
    r"""|Δ| between run 1 and run 0.   xs, vs: (T, K, N, 2) → (T,)"""
    dx, dv = xs[:, 1] - xs[:, 0], vs[:, 1] - vs[:, 0]  # (T, N, 2)
    return (torch.einsum("tai,tai->t", dx, dx) + torch.einsum("tai,tai->t", dv, dv)).sqrt()
