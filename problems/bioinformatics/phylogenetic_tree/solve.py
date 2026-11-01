"""Distances → trees (UPGMA and neighbour joining) and the Robinson–Foulds comparison of tree shapes."""

import torch

from .equation import jukes_cantor, nj_criterion, p_distance
from .tensors import Tree


def distances(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """p-distance and Jukes–Cantor distance between all leaves. X: (n, L, 4) → two (n, n) matrices."""
    p = p_distance(X)
    d = jukes_cantor(p)
    return p, d.fill_diagonal_(0.0)


def _closest_pair(score: torch.Tensor, active: list[int]) -> tuple[int, int]:
    """Indices (into the full matrix) of the active pair with the smallest score."""
    sub = score[active][:, active].clone().fill_diagonal_(float("inf"))
    k = int(sub.argmin())
    return active[k // len(active)], active[k % len(active)]


def upgma(D: torch.Tensor) -> Tree:
    """Average-linkage clustering: merge the closest clusters, place them at height d/2 (a molecular clock)."""
    n = D.shape[0]
    D = D.clone()
    node = list(range(n))  # tree node held in each row of D
    size, height = [1] * n, [0.0] * n
    parent, length = [-1] * (2 * n - 1), [0.0] * (2 * n - 1)
    active = list(range(n))
    for new in range(n, 2 * n - 1):
        i, j = _closest_pair(D, active)
        h = float(D[i, j]) / 2
        for row in (i, j):
            parent[node[row]], length[node[row]] = new, max(0.0, h - height[row])
        merged = (size[i] * D[i] + size[j] * D[j]) / (size[i] + size[j])  # size-weighted average distance
        D[i], D[:, i] = merged, merged
        node[i], size[i], height[i] = new, size[i] + size[j], h
        active.remove(j)
    return Tree(n, parent, length)


def neighbour_joining(D: torch.Tensor) -> Tree:
    """Saitou–Nei: merge the pair minimising Q, give each its branch length, replace both by the new node u."""
    n = D.shape[0]
    D = D.clone()
    node = list(range(n))
    parent, length = [-1] * (2 * n - 2), [0.0] * (2 * n - 2)
    active = list(range(n))
    for new in range(n, 2 * n - 2):
        sub = D[active][:, active]
        Q = nj_criterion(sub)
        k = int(Q.argmin())
        a, b = k // len(active), k % len(active)
        i, j = active[a], active[b]
        r = sub.sum(dim=1)  # r_i = D_{ij} 1_j over the active nodes
        li = 0.5 * float(D[i, j]) + float(r[a] - r[b]) / (2 * (len(active) - 2))
        lj = float(D[i, j]) - li
        parent[node[i]], length[node[i]] = new, max(0.0, li)
        parent[node[j]], length[node[j]] = new, max(0.0, lj)
        du = 0.5 * (D[i] + D[j] - D[i, j])  # distance of the new node u to every other node
        D[i], D[:, i] = du, du
        D[i, i] = 0.0
        node[i] = new
        active.remove(j)
        if len(active) == 2:  # the last two nodes are joined by one branch: hang the leftover under u
            other = active[0] if active[1] == i else active[1]
            parent[node[other]], length[node[other]] = new, max(0.0, float(D[i, other]))
            break
    return Tree(n, parent, length)


def midpoint_root(tree: Tree) -> Tree:
    """Re-root an unrooted tree at the middle of its longest leaf-to-leaf path (for drawing only).

    The branch holding the midpoint is cut in two by a new root node; then the parent links are re-derived by a
    breadth-first walk from it. Leaves keep their numbers, so the splits do not change.
    """
    edges: dict[int, dict[int, float]] = {v: {} for v in range(len(tree.parent))}
    for v, p in enumerate(tree.parent):
        if p >= 0:
            edges[v][p] = edges[p][v] = tree.length[v]

    def walk(start: int):  # distance and previous node of every node, seen from `start`
        dist, prev, todo = {start: 0.0}, {start: -1}, [start]
        for u in todo:
            for w, ell in edges[u].items():
                if w not in dist:
                    dist[w], prev[w] = dist[u] + ell, u
                    todo.append(w)
        return dist, prev

    d0, _ = walk(0)
    a = max(range(tree.n_leaves), key=lambda v: d0[v])  # one end of the longest path
    da, prev = walk(a)
    b = max(range(tree.n_leaves), key=lambda v: da[v])  # the other end
    half, v = da[b] / 2, b
    while da[prev[v]] > half:  # climb from b towards a until the branch (prev[v], v) holds the midpoint
        v = prev[v]
    u, root = prev[v], len(tree.parent)
    del edges[u][v], edges[v][u]
    edges[root] = {u: half - da[u], v: da[v] - half}
    edges[u][root], edges[v][root] = edges[root][u], edges[root][v]
    _, parent = walk(root)
    length = [0.0 if parent[w] < 0 else edges[w][parent[w]] for w in range(root + 1)]
    return Tree(tree.n_leaves, [parent[w] for w in range(root + 1)], length)


def clades(tree: Tree) -> list[frozenset]:
    """Leaf set below every node (index = node)."""
    below: list[set] = [set() for _ in tree.parent]
    for leaf in range(tree.n_leaves):
        v = leaf
        while v >= 0:
            below[v].add(leaf)
            v = tree.parent[v]
    return [frozenset(s) for s in below]


def splits(tree: Tree) -> set[frozenset]:
    """Non-trivial bipartitions of the leaves made by cutting one branch, written as the side without leaf 0.

    The root is ignored, so rooted and unrooted trees with the same shape have the same splits.
    """
    n, out = tree.n_leaves, set()
    for clade in clades(tree):
        side = clade if 0 not in clade else frozenset(range(n)) - clade
        if 2 <= len(side) <= n - 2:
            out.add(side)
    return out


def robinson_foulds(a: Tree, b: Tree) -> int:
    """Number of splits found in only one of the two trees (0 = same unrooted topology)."""
    return len(splits(a) ^ splits(b))
