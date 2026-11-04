r"""Flocking rules → tensor translation.

Every bird interacts with the birds within a radius R in a periodic box of side L. All N² displacements are one
tensor, folded back with the minimum-image convention (the nearest copy of bird j):

    \mathbf d_{ij} = (\mathbf x_j - \mathbf x_i) - L\,\operatorname{round}\!\big((\mathbf x_j - \mathbf x_i)/L\big),
    \qquad A_{ij} = [\,|\mathbf d_{ij}| < R\,].

Vicsek model (Vicsek et al. 1995). Each bird flies at constant speed v_0 and takes the average heading of its
neighbours (itself included) plus a random angle in [−η/2, η/2]:

    \theta_i \leftarrow \arg\Big(\sum_j A_{ij}\,e^{i\theta_j}\Big) + \eta\,\xi_i,\qquad
    \mathbf x_i \leftarrow \mathbf x_i + v_0\,(\cos\theta_i, \sin\theta_i).

Boids (Reynolds 1987): separation, alignment and cohesion, here written as pairwise forces

    \mathbf a_i = \sum_j \Big[-w_s S_{ij}\frac{\mathbf d_{ij}}{|\mathbf d_{ij}|^2}
                  + w_a A_{ij}(\mathbf v_j - \mathbf v_i) + w_c A_{ij}\,\mathbf d_{ij}\Big],

each antisymmetric in i ↔ j, so Σ_i a_i = 0 (Newton's third law) until the speed is renormalised to v_0.
The polar order parameter φ = |Σ_i v_i| / Σ_i |v_i| is 1 for a flock moving together and ~1/√N for disorder.
"""

import torch


def minimum_image(d: torch.Tensor, box: float) -> torch.Tensor:
    """Fold a displacement into [−L/2, L/2] in each direction: the nearest periodic copy.   (...) → (...)"""
    return d - box * torch.round(d / box)


def pair_displacements(x: torch.Tensor, box: float) -> torch.Tensor:
    """d_ij = x_j − x_i with the minimum image.   (B, N, 2) → (B, N, N, 2)"""
    return minimum_image(x[:, None, :, :] - x[:, :, None, :], box)


def neighbours(d: torch.Tensor, radius: float) -> torch.Tensor:
    """A_ij = 1 if |d_ij| < R (bird i itself included).   (B, N, N, 2) → (B, N, N) float"""
    return (d.pow(2).sum(-1) < radius**2).to(d.dtype)


def polar_order(v: torch.Tensor) -> torch.Tensor:
    """φ = |Σ_i v_i| / Σ_i |v_i|   (i summed).   (B, N, 2) → (B,)"""
    return v.sum(1).norm(dim=-1) / v.norm(dim=-1).sum(1)


def vicsek_headings(x: torch.Tensor, theta: torch.Tensor, box: float, radius: float) -> torch.Tensor:
    r"""Mean heading of the neighbours: arg Σ_j A_ij e^{iθ_j}   (j summed).   (B, N, 2), (B, N) → (B, N)"""
    a = neighbours(pair_displacements(x, box), radius)  # (B, N, N)
    z = torch.einsum("bij,bj->bi", a.to(torch.complex64), torch.exp(1j * theta).to(torch.complex64))
    return z.angle().to(theta.dtype)


def vicsek_step(
    x: torch.Tensor, theta: torch.Tensor, eta: torch.Tensor, xi: torch.Tensor, box: float, radius: float, v0: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Vicsek update for every noise level at once: η (B, 1), ξ uniform in [−½, ½] (B, N).
    (B, N, 2), (B, N) → new positions (B, N, 2) in [0, L), new headings (B, N)"""
    theta = vicsek_headings(x, theta, box, radius) + eta * xi
    step = v0 * torch.stack([torch.cos(theta), torch.sin(theta)], -1)  # (B, N, 2) constant speed v0
    return torch.remainder(x + step, box), theta


def reynolds_acceleration(x: torch.Tensor, v: torch.Tensor, box: float, radius: float, w: dict) -> torch.Tensor:
    """Separation (within R/4), alignment and cohesion (within R) as antisymmetric pair forces.
    (B, N, 2), (B, N, 2) → (B, N, 2)"""
    d = pair_displacements(x, box)  # (B, N, N, 2)
    a = neighbours(d, radius)  # (B, N, N)
    r2 = d.pow(2).sum(-1)
    close = neighbours(d, radius / 4) * (r2 > 0)  # bird i itself excluded
    repel = torch.einsum("bij,bijk->bik", close / r2.clamp(min=1e-12), d)  # Σ_j S_ij d_ij / |d_ij|²
    align = torch.einsum("bij,bjk->bik", a, v) - a.sum(-1, keepdim=True) * v  # Σ_j A_ij (v_j − v_i)
    cohere = torch.einsum("bij,bijk->bik", a, d)  # Σ_j A_ij d_ij: pull towards the local centre
    return -w["separation"] * repel + w["alignment"] * align + w["cohesion"] * cohere
