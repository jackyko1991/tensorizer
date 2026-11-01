"""Build the concrete tensors: a random true tree and DNA sequences evolved along it (synthetic, seeded)."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .equation import evolve_branch, jc_transition


@dataclass
class Tree:
    """Leaves are nodes 0 … n−1. parent[v] is v's parent (−1 for the root), length[v] the branch above v."""

    n_leaves: int
    parent: list[int]
    length: list[float]

    @property
    def root(self) -> int:
        return self.parent.index(-1)

    def children(self) -> list[list[int]]:
        kids: list[list[int]] = [[] for _ in self.parent]
        for v, p in enumerate(self.parent):
            if p >= 0:
                kids[p].append(v)
        return kids


def random_tree(n_taxa: int, depth: float, clock: bool, seed: int) -> Tree:
    """Coalescent tree: join two random lineages at exponentially spaced times, rescaled to root height `depth`.

    Without a clock each branch is then stretched by a random rate factor (log-normal), so leaves end up at
    different distances from the root (non-ultrametric).
    """
    gen = torch.Generator().manual_seed(seed)
    height = [0.0] * n_taxa
    parent = [-1] * (2 * n_taxa - 1)
    active, t = list(range(n_taxa)), 0.0
    while len(active) > 1:
        k = len(active)
        t += float(torch.empty(1).exponential_(k * (k - 1) / 2, generator=gen))
        pick = torch.randperm(k, generator=gen)[:2].tolist()
        a, b = active[pick[0]], active[pick[1]]
        new = len(height)
        height.append(t)
        parent[a] = parent[b] = new
        active = [v for v in active if v not in (a, b)] + [new]
    scale = depth / height[-1]
    length = [0.0 if p < 0 else (height[p] - height[v]) * scale for v, p in enumerate(parent)]
    if not clock:
        rates = torch.exp(0.7 * torch.randn(len(parent), generator=gen))
        length = [ell * float(r) for ell, r in zip(length, rates)]
    return Tree(n_taxa, parent, length)


def path_matrix(tree: Tree, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """M_{iv} = 1 if branch v is on the path from leaf i to the root: (n, V); branch lengths b_v: (V,)."""
    M = torch.zeros(tree.n_leaves, len(tree.parent), device=device)
    for i in range(tree.n_leaves):
        v = i
        while tree.parent[v] >= 0:
            M[i, v] = 1.0
            v = tree.parent[v]
    return M, torch.tensor(tree.length, device=device)


def simulate(tree: Tree, n_sites: int, n_rep: int, seed: int, device: torch.device) -> torch.Tensor:
    """Evolve R independent alignments down the tree under Jukes–Cantor: leaf sequences one-hot, (R, n, L, 4).

    Nodes are created bottom-up, so going through them in reverse order visits every parent before its children.
    """
    gen = torch.Generator().manual_seed(seed)
    seqs: dict[int, torch.Tensor] = {}
    root = tree.root
    seqs[root] = F.one_hot(torch.randint(0, 4, (n_rep, n_sites), generator=gen), 4).float().to(device)
    for v in range(len(tree.parent) - 1, -1, -1):
        if v == root:
            continue
        probs = evolve_branch(seqs[tree.parent[v]], jc_transition(tree.length[v], device))  # (R, L, 4)
        u = torch.rand(n_rep, n_sites, 1, generator=gen).to(device)
        letter = (u > probs.cumsum(dim=-1)).sum(dim=-1).clamp(max=3)  # inverse-CDF sampling at every site
        seqs[v] = F.one_hot(letter, 4).float()
    return torch.stack([seqs[i] for i in range(tree.n_leaves)], dim=1)
