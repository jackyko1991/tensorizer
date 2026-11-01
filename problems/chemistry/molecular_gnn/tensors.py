"""Build the concrete tensors: random conjugated hydrocarbon graphs (padded to N atoms), atom features, weights.

A molecule is its hydrogen-suppressed carbon skeleton: atoms are sp² carbons (one p orbital each), bonds are
C–C σ bonds. Skeletons are grown at random as trees with at most 3 neighbours per carbon, then closed into
5- and 6-membered rings. They are graph-theoretic toys, not a curated set of real molecules.

Padding: a batch of molecules with up to N atoms is stored as
    A (M, N, N)  adjacency, 0 for padded atoms         mask (M, N)  1 for real atoms, 0 for padding
    x (M, N, F)  atom features (one-hot number of carbon neighbours 1, 2, 3), 0 for padding
"""

import math
import random

import torch

MAX_DEGREE = 3  # an sp² carbon has three σ bonds (to C or H)
N_FEATURES = MAX_DEGREE


def _distance(adj: list[set[int]], start: int) -> list[int]:
    """Breadth-first graph distances from one atom."""
    dist = [-1] * len(adj)
    dist[start], frontier = 0, [start]
    while frontier:
        nxt = []
        for i in frontier:
            for j in adj[i]:
                if dist[j] < 0:
                    dist[j] = dist[i] + 1
                    nxt.append(j)
        frontier = nxt
    return dist


def random_skeleton(n: int, rng: random.Random) -> list[set[int]]:
    """A random connected carbon skeleton with n atoms: a tree with degree ≤ 3, plus ring closures that make
    5- or 6-membered rings (bonds between atoms at graph distance 4 or 5)."""
    adj = [set() for _ in range(n)]
    for new in range(1, n):  # attach each new atom to an earlier atom that still has a free valence
        free = [i for i in range(new) if len(adj[i]) < MAX_DEGREE]
        parent = rng.choice(free)
        adj[new].add(parent)
        adj[parent].add(new)
    for _ in range(rng.randint(0, n // 4)):  # a few ring closures
        free = [i for i in range(n) if len(adj[i]) < MAX_DEGREE]
        pairs = [(i, j) for i in free for j in free if i < j and _distance(adj, i)[j] in (4, 5)]
        if not pairs:
            break
        i, j = rng.choice(pairs)
        adj[i].add(j)
        adj[j].add(i)
    return adj


def dataset(n_mol: int, sizes: tuple[int, int], n_pad: int, seed: int, device: torch.device) -> dict:
    """n_mol random molecules with sizes[0] … sizes[1] atoms, padded to n_pad: A (M, N, N), mask (M, N), x (M, N, F)."""
    rng = random.Random(seed)
    A = torch.zeros(n_mol, n_pad, n_pad, dtype=torch.float32)
    for m in range(n_mol):
        n = rng.randint(*sizes)
        for i, nbrs in enumerate(random_skeleton(n, rng)):
            A[m, i, list(nbrs)] = 1.0
    mask = (A.sum(-1) > 0).to(torch.float32)  # every real atom has at least one bond (n ≥ 2)
    return {"A": A.to(device), "mask": mask.to(device), "x": features(A, mask).to(device)}


def features(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x_{mif} = 1 if atom i has f+1 carbon neighbours (f = 0, 1, 2): (M, N, N) → (M, N, F)."""
    degree = A.sum(-1).long()  # (M, N)
    x = torch.nn.functional.one_hot((degree - 1).clamp(0, N_FEATURES - 1), N_FEATURES).to(A.dtype)
    return x * mask[..., None]


def permute(data: dict, perm: torch.Tensor) -> dict:
    """Relabel the atoms of every molecule: A' = Π A Πᵀ, x' = Π x, mask' = Π mask (same permutation for all)."""
    return {"A": data["A"][:, perm][:, :, perm], "mask": data["mask"][:, perm], "x": data["x"][:, perm]}


def init_gnn(width: int, n_layers: int, seed: int, device: torch.device) -> dict:
    """GNN weights: embed W_in (F, C); per layer U[l], V[l] (C, C), c[l] (C,); readout Q1 (C, C), Q2 (C,)."""
    gen = torch.Generator().manual_seed(seed)

    def dense(fan_in, *shape):
        return (torch.rand(*shape, generator=gen, dtype=torch.float32) * 2 - 1) / math.sqrt(fan_in)

    w = {
        "W_in": dense(N_FEATURES, N_FEATURES, width),
        "b_in": torch.zeros(width, dtype=torch.float32),
        "U": [dense(width, width, width) for _ in range(n_layers)],
        "V": [dense(width, width, width) * 0.5 for _ in range(n_layers)],
        "c": [torch.zeros(width, dtype=torch.float32) for _ in range(n_layers)],
        "Q1": dense(width, width, width),
        "q1": torch.zeros(width, dtype=torch.float32),
        "Q2": dense(width, width),
        "q2": torch.zeros((), dtype=torch.float32),
    }
    return _trainable(w, device)


def init_mlp(n_pad: int, hidden: int, seed: int, device: torch.device) -> dict:
    """Baseline MLP on the flattened padded adjacency: (N², H), (H, H), (H,) weights."""
    gen = torch.Generator().manual_seed(seed + 1)

    def dense(fan_in, *shape):
        return (torch.rand(*shape, generator=gen, dtype=torch.float32) * 2 - 1) / math.sqrt(fan_in)

    w = {
        "W1": dense(n_pad * n_pad, n_pad * n_pad, hidden),
        "b1": torch.zeros(hidden, dtype=torch.float32),
        "W2": dense(hidden, hidden, hidden),
        "b2": torch.zeros(hidden, dtype=torch.float32),
        "W3": dense(hidden, hidden),
        "b3": torch.zeros((), dtype=torch.float32),
    }
    return _trainable(w, device)


def _trainable(w: dict, device: torch.device) -> dict:
    return {
        k: [t.to(device).requires_grad_(True) for t in v] if isinstance(v, list) else v.to(device).requires_grad_(True)
        for k, v in w.items()
    }


def parameters(w: dict) -> list[torch.Tensor]:
    return [t for v in w.values() for t in (v if isinstance(v, list) else [v])]


def count(w: dict) -> int:
    return sum(t.numel() for t in parameters(w))
