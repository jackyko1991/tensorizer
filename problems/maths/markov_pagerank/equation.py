r"""Markov chain on a web graph → tensor translation.

A random surfer on page j clicks one of its d_j = \sum_i L_{ij} outgoing links at random (i summed), so the
transition matrix is column-stochastic,

    S_{ij} = L_{ij} / d_j,       and S_{ij} = 1/N for a dangling page (d_j = 0: jump anywhere).

With probability 1 − d the surfer instead teleports to a uniformly random page (Google matrix),

    G_{ij} = d\,S_{ij} + (1 - d)/N,

and the probability of being on each page evolves as x_i \leftarrow G_{ij} x_j (j summed). PageRank is the
stationary distribution: G_{ij} \pi_j = \pi_i, \sum_i \pi_i = 1, the eigenvector with eigenvalue 1.
"""

import torch


def transition(links: torch.Tensor) -> torch.Tensor:
    r"""S_{ij} = L_{ij} / d_j with d_j = Σ_i L_{ij}; dangling columns become uniform.   (N, N) → (N, N)"""
    n = links.shape[0]
    out_degree = torch.einsum("ij->j", links)  # (N,)
    s = links / out_degree.clamp(min=1)
    return torch.where(out_degree == 0, 1.0 / n, s)  # broadcasts the (N,) mask along the columns


def google(s: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    r"""G_{bij} = d_b S_{ij} + (1 − d_b)/N for a batch of damping factors d_b.   (N, N), (B,) → (B, N, N)"""
    n = s.shape[0]
    return d[:, None, None] * s + (1 - d[:, None, None]) / n


def power_step(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""x'_{bi} = G_{bij} x_{bj}   (j summed).   (B, N, N), (B, N) → (B, N)"""
    return torch.einsum("bij,bj->bi", g, x)


def walk_step(pos: torch.Tensor, s: torch.Tensor, d: float, gen: torch.Generator) -> torch.Tensor:
    """Move W surfers at once: follow a link (column pos of S) with probability d, else teleport.   (W,) → (W,)"""
    n = s.shape[0]
    follow = torch.multinomial(s.T[pos], 1, generator=gen)[:, 0]  # (W,): row w of S.T[pos] is page pos_w's links
    jump = torch.randint(0, n, pos.shape, generator=gen)
    teleport = torch.rand(pos.shape, generator=gen) > d
    return torch.where(teleport, jump, follow)
