"""Build the concrete tensors: contact networks as dense adjacency matrices A_{ij} (N, N), symmetric 0/1.

- load_workplace: a real face-to-face contact network (SocioPatterns, office building in France, 2013; CC0)
- erdos_renyi: every pair linked independently with the same probability (a Poisson degree distribution)
- barabasi_albert: preferential attachment, a scale-free network with a few big hubs (P(k) ∝ k^-3)
- complete_graph, ring_lattice: the well-mixed limit and a k-regular test case
"""

from pathlib import Path

import torch

DATA = Path(__file__).parent / "data" / "workplace_invs13.csv"


def from_edges(i: torch.Tensor, j: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Symmetric 0/1 adjacency (N, N) from edge lists i, j (no self-loops)."""
    A = torch.zeros(n, n, device=device)
    A[i, j] = 1.0
    A[j, i] = 1.0
    A.fill_diagonal_(0.0)
    return A


def load_workplace(device: torch.device, min_windows: int = 1) -> torch.Tensor:
    """The 92 workers; an edge joins two people seen face to face in at least `min_windows` 20-second windows."""
    rows = [line.split(",") for line in DATA.read_text().splitlines()[1:]]
    e = torch.tensor([[int(a), int(b), int(w)] for a, b, w in rows])
    keep = e[:, 2] >= min_windows
    n = int(e[:, :2].max()) + 1
    return from_edges(e[keep, 0], e[keep, 1], n, device)


def erdos_renyi(n: int, mean_degree: float, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """G(N, p) with p = <k> / (N − 1): one coin flip per pair (upper triangle), mirrored."""
    p = mean_degree / (n - 1)
    upper = torch.triu(torch.rand(n, n, generator=gen) < p, diagonal=1).float()
    return (upper + upper.T).to(device)


def barabasi_albert(n: int, m: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """Preferential attachment: start from m + 1 fully linked nodes; each new node links to m distinct old nodes
    chosen with probability ∝ their degree. Growth is sequential by definition, so this is a loop over new nodes."""
    A = torch.zeros(n, n)
    A[: m + 1, : m + 1] = 1.0 - torch.eye(m + 1)
    deg = A.sum(1)
    for new in range(m + 1, n):
        targets = torch.multinomial(deg[:new], m, replacement=False, generator=gen)
        A[new, targets] = A[targets, new] = 1.0
        deg[targets] += 1
        deg[new] = m
    return A.to(device)


def complete_graph(n: int, device: torch.device) -> torch.Tensor:
    """Everybody meets everybody: the well-mixed population of the classic SIR model."""
    return torch.ones(n, n, device=device) - torch.eye(n, device=device)


def ring_lattice(n: int, k: int, device: torch.device) -> torch.Tensor:
    """N nodes on a circle, each linked to its k/2 nearest neighbours on either side: every degree is k."""
    idx = torch.arange(n, device=device)
    gap = (idx[:, None] - idx[None, :]).abs()
    gap = torch.minimum(gap, n - gap)  # distance round the ring
    return ((gap >= 1) & (gap <= k // 2)).float()


def layout(A: torch.Tensor, n_iter: int = 300, seed: int = 0) -> torch.Tensor:
    """2D positions (N, 2) for drawing, by a force-directed layout (Fruchterman–Reingold) on all pairs at once:
    linked nodes attract like springs, every pair repels."""
    gen = torch.Generator().manual_seed(seed)
    n = A.shape[0]
    x = torch.randn(n, 2, generator=gen, dtype=torch.float64)
    A = A.cpu().double()
    k = 1.0 / n**0.5
    for it in range(n_iter):
        d = x[:, None, :] - x[None, :, :]  # (N, N, 2) separation vectors
        r = d.norm(dim=-1).clamp_min(1e-3)  # (N, N)
        force = (k**2 / r**2 - A * r / k)[..., None] * d  # repulsion − attraction along d
        step = force.sum(1)
        temp = 0.1 * (1 - it / n_iter) + 1e-3
        x = x + temp * step / step.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x = x - x.mean(0)
    return (x / x.abs().max()).float()
