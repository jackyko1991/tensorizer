"""Build the concrete tensors: node positions, the spring list, the incidence matrix and the pins.

Node (row r, column c) of an n × n grid has index a = r·n + c and starts flat at (c·h, r·h, 0).
float64 keeps the energy bookkeeping clean.
"""

import torch

DTYPE = torch.float64


def grid(n: int, spacing: float, device: torch.device | None = None) -> torch.Tensor:
    """x_ai of shape (N = n², 3): a flat horizontal sheet."""
    r, c = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")  # (n, n) each
    x = torch.stack([c * spacing, r * spacing, torch.zeros(n, n)], dim=-1)  # (n, n, 3)
    return x.reshape(n * n, 3).to(DTYPE).to(device)


def springs(n: int, device: torch.device | None = None) -> torch.Tensor:
    """Edge list (E, 2) of node pairs: structural (neighbours), shear (diagonals) and bend (two apart) springs."""
    ids = torch.arange(n * n).reshape(n, n)  # ids[r, c] = r·n + c
    pairs = [
        (ids[:, :-1], ids[:, 1:]),  # horizontal neighbours
        (ids[:-1, :], ids[1:, :]),  # vertical neighbours
        (ids[:-1, :-1], ids[1:, 1:]),  # diagonal ↘
        (ids[:-1, 1:], ids[1:, :-1]),  # diagonal ↙ (stops the cloth shearing flat)
        (ids[:, :-2], ids[:, 2:]),  # two apart, horizontally (resists bending)
        (ids[:-2, :], ids[2:, :]),  # two apart, vertically
    ]
    edges = torch.cat([torch.stack([a.flatten(), b.flatten()], dim=-1) for a, b in pairs])
    return edges.to(device)


def incidence(edges: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """D_ea of shape (E, N): −1 at the first node of spring e, +1 at the second, 0 elsewhere."""
    D = torch.zeros(len(edges), n_nodes, dtype=DTYPE, device=edges.device)
    e = torch.arange(len(edges), device=edges.device)
    D[e, edges[:, 0]] = -1.0
    D[e, edges[:, 1]] = 1.0
    return D


def rest_lengths(x0: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """ℓ⁰_e of shape (E,): every spring is relaxed in the starting configuration."""
    return (x0[edges[:, 1]] - x0[edges[:, 0]]).norm(dim=-1)


def free_mask(n: int, device: torch.device | None = None) -> torch.Tensor:
    """(N,) with 1 for moving nodes and 0 for the two pinned corners of the first row."""
    free = torch.ones(n * n, dtype=DTYPE, device=device)
    free[0] = 0.0  # corner (0, 0)
    free[n - 1] = 0.0  # corner (0, n − 1)
    return free


def parameters(k: float, gamma: float, node_mass: float, g: float, device: torch.device | None = None) -> dict:
    """Material constants in one dict; g_i points down the z axis."""
    gravity = torch.tensor([0.0, 0.0, -g], dtype=DTYPE, device=device)  # (3,)
    return {"k": k, "gamma": gamma, "m": node_mass, "g": gravity}
