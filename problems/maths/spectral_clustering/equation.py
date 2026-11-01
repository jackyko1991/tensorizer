r"""Mathematics → tensor translation.

Points x_{ia} (i = point, a = coordinate) become a weighted graph:

    D2_{ij} = (x_{ia} − x_{ja})(x_{ia} − x_{ja})               a summed: squared distances
    W_{ij}  = exp(−D2_{ij} / 2σ²)   on k-nearest-neighbour edges, 0 elsewhere (and on the diagonal)
    d_i     = W_{ij} 1_j                                         j summed: the degree of point i
    L_{ij}  = d_i δ_{ij} − W_{ij}                                graph Laplacian L = D − W
    (L_sym)_{ij} = δ_{ij} − d_i^{−1/2} W_{ij} d_j^{−1/2}         normalised Laplacian

The key identity, with i and j summed,

    f_i L_{ij} f_j = ½ W_{ij} (f_i − f_j)²  ≥ 0,

says L is positive semi-definite, and f^T L f = 0 exactly when f is constant on every connected piece of the graph.
So the number of zero eigenvalues of L is the number of connected components.
"""

import torch


def squared_distances(x: torch.Tensor) -> torch.Tensor:
    r"""D2_{ij} = x_{ia}x_{ia} + x_{ja}x_{ja} − 2 x_{ia}x_{ja}   (a summed).   (N, 2) → (N, N)"""
    sq = torch.einsum("ia,ia->i", x, x)
    return (sq[:, None] + sq[None, :] - 2 * torch.einsum("ia,ja->ij", x, x)).clamp_min(0)


def gaussian_knn_weights(d2: torch.Tensor, sigma: float, k: int) -> torch.Tensor:
    r"""W_{ij} = exp(−D2_{ij}/2σ²) if j is among the k nearest neighbours of i or i among those of j.   (N, N)"""
    n = d2.shape[0]
    near = torch.zeros_like(d2, dtype=torch.bool)
    nearest = d2.topk(k + 1, dim=1, largest=False).indices  # (N, k+1), includes the point itself
    near[torch.arange(n, device=d2.device)[:, None], nearest] = True
    near = near | near.T  # symmetric graph
    near.fill_diagonal_(False)
    return torch.where(near, torch.exp(-d2 / (2 * sigma**2)), 0.0)


def degrees(w: torch.Tensor) -> torch.Tensor:
    r"""d_i = W_{ij} 1_j   (j summed).   (N, N) → (N,)"""
    return torch.einsum("ij->i", w)


def laplacian(w: torch.Tensor) -> torch.Tensor:
    r"""L_{ij} = d_i δ_{ij} − W_{ij}.   (N, N)"""
    return torch.diag(degrees(w)) - w


def normalized_laplacian(w: torch.Tensor) -> torch.Tensor:
    r"""(L_sym)_{ij} = δ_{ij} − d_i^{−1/2} W_{ij} d_j^{−1/2}   (no sum: i and j appear on both sides).   (N, N)"""
    s = degrees(w).clamp_min(1e-12).rsqrt()
    return torch.eye(w.shape[0], dtype=w.dtype, device=w.device) - torch.einsum("i,ij,j->ij", s, w, s)


def dirichlet_energy(w: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    r"""½ W_{ij} (f_i − f_j)²   (i, j summed): equals f_i L_{ij} f_j and can never be negative.   f: (N,) → ()"""
    return 0.5 * torch.einsum("ij,ij->", w, (f[:, None] - f[None, :]) ** 2)
