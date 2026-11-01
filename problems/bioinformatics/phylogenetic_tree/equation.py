r"""Molecular evolution → tensor translation.

Jukes–Cantor model: every site mutates at the same rate to any of the other three letters. After a branch of length
t (expected substitutions per site) the letter-to-letter transition matrix is

    P_{\alpha\beta}(t) = 1/4 + (\delta_{\alpha\beta} − 1/4)\,e^{-4t/3}

so a child's letter probabilities are y_{l\beta} = x_{l\alpha} P_{\alpha\beta} (\alpha summed) for every site l at once.

Distances between n aligned sequences with one-hot letters X_{il\alpha} (taxon i, site l, letter \alpha):

    p_{ij} = 1 − X_{il\alpha} X_{jl\alpha} / L                 (l, \alpha summed: fraction of sites that differ)
    d_{ij} = −(3/4) \ln(1 − 4 p_{ij} / 3)                      Jukes–Cantor correction for hidden multiple hits

d estimates the path length between i and j on the tree; p saturates at 3/4 and underestimates long distances.

Tree building from d: UPGMA merges the closest pair and averages distances (assumes a molecular clock); neighbour
joining merges the pair minimising Q_{ij} = (n − 2) d_{ij} − r_i − r_j with r_i = d_{ij} 1_j (j summed), which
recovers the right tree for any additive distances, clock or not.
"""

import torch


def jc_transition(t: float, device: torch.device) -> torch.Tensor:
    r"""P_{αβ}(t) = 1/4 + (δ_{αβ} − 1/4) e^{−4t/3}: (4, 4), rows sum to 1."""
    eye = torch.eye(4, device=device)
    return 0.25 + (eye - 0.25) * torch.exp(torch.tensor(-4.0 * t / 3.0, device=device))


def evolve_branch(x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    r"""y_{rlβ} = x_{rlα} P_{αβ}   (α summed): letter probabilities of the child at every site.   (R, L, 4)"""
    return torch.einsum("rla,ab->rlb", x, P)


def p_distance(X: torch.Tensor) -> torch.Tensor:
    r"""p_{ij} = 1 − X_{ilα} X_{jlα} / L   (l, α summed).   (..., n, L, 4) → (..., n, n)"""
    return 1.0 - torch.einsum("...ila,...jla->...ij", X, X) / X.shape[-2]


def jukes_cantor(p: torch.Tensor) -> torch.Tensor:
    r"""d_{ij} = −(3/4) ln(1 − 4 p_{ij}/3); p ≥ 3/4 (saturated) is capped just below."""
    return -0.75 * torch.log1p(-4.0 / 3.0 * p.clamp(max=0.749))


def nj_criterion(D: torch.Tensor) -> torch.Tensor:
    r"""Q_{ij} = (n − 2) D_{ij} − r_i − r_j with r_i = D_{ij} 1_j (j summed); diagonal set to +∞.   (n, n) → (n, n)"""
    n = D.shape[0]
    r = D.sum(dim=1)
    Q = (n - 2) * D - r[:, None] - r[None, :]
    return Q.fill_diagonal_(float("inf"))


def patristic(M: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Path lengths on a tree: d_{ij} = M_{iv} b_v + M_{jv} b_v − 2 M_{iv} b_v M_{jv}   (v summed over branches).

    M_{iv} = 1 when branch v (the edge above node v) lies on the path from leaf i to the root; b_v its length.
    (n, V), (V,) → (n, n)
    """
    depth = M @ b  # (n,) distance of every leaf from the root
    shared = torch.einsum("iv,v,jv->ij", M, b, M)  # length of the common root path of i and j
    return depth[:, None] + depth[None, :] - 2 * shared
