"""Build the concrete tensors: three synthetic point clouds with true labels, and a 2D grid graph."""

import math

import torch

DATASETS = ["moons", "circles", "blobs"]
N_CLUSTERS = {"moons": 2, "circles": 2, "blobs": 3}


def points(name: str, n: int, noise: float, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Point cloud x_{ia} of shape (N, 2), float64, and its true cluster labels (N,)."""
    gen = torch.Generator().manual_seed(seed)
    k = N_CLUSTERS[name]
    labels = torch.arange(n) % k
    if name == "moons":  # two interleaved half circles
        t = torch.rand(n, generator=gen, dtype=torch.float64) * math.pi
        upper = torch.stack([torch.cos(t), torch.sin(t)], 1)
        lower = torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)
        x = torch.where(labels[:, None] == 0, upper, lower)
    elif name == "circles":  # a ring inside a ring
        t = torch.rand(n, generator=gen, dtype=torch.float64) * 2 * math.pi
        radius = torch.where(labels == 0, 1.0, 0.45).double()
        x = radius[:, None] * torch.stack([torch.cos(t), torch.sin(t)], 1)
    else:  # three round blobs
        centres = torch.tensor([[0.0, 0.0], [2.2, 0.4], [0.9, 2.0]], dtype=torch.float64)
        x = centres[labels] + 4 * noise * torch.randn(n, 2, generator=gen, dtype=torch.float64)
    x = x + noise * torch.randn(n, 2, generator=gen, dtype=torch.float64)
    return x.to(device), labels.to(device)


def grid_graph(n: int, device: torch.device) -> torch.Tensor:
    """Adjacency of an n×n grid (4 neighbours, free edges): W = A₁ ⊗ I + I ⊗ A₁ with A₁ the path graph.   (n², n²)"""
    a1 = torch.diag(torch.ones(n - 1, dtype=torch.float64), 1)
    a1 = a1 + a1.T  # path graph 0 – 1 – … – (n−1)
    eye = torch.eye(n, dtype=torch.float64)
    return (torch.kron(a1, eye) + torch.kron(eye, a1)).to(device)
