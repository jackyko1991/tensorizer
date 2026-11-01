"""Eigendecomposition of the Laplacian, k-means, spectral clustering and the adjusted Rand index."""

import torch

from .equation import degrees, normalized_laplacian


def spectrum(lap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigenvalues in ascending order (N,) and orthonormal eigenvectors in the columns (N, N)."""
    return torch.linalg.eigh(lap)


def count_zero_eigenvalues(evals: torch.Tensor, tol: float = 1e-8) -> int:
    return int((evals.abs() < tol).sum())


def connected_components(w: torch.Tensor) -> torch.Tensor:
    """Component label of every node by label propagation: each node takes the smallest label among its neighbours
    until nothing changes. Independent of the eigenvalues, so it checks them.   (N, N) → (N,)"""
    n = w.shape[0]
    linked = (w > 0) | torch.eye(n, dtype=torch.bool, device=w.device)
    label = torch.arange(n, device=w.device, dtype=torch.float64)
    while True:
        new = torch.where(linked, label[None, :], torch.inf).min(dim=1).values
        if torch.equal(new, label):
            return label.long()
        label = new


def kmeans(x: torch.Tensor, k: int, seed: int, n_init: int = 10, n_iter: int = 100) -> torch.Tensor:
    """Lloyd's algorithm with k-means++ starts; keeps the run with the smallest within-cluster sum of squares."""
    gen = torch.Generator().manual_seed(seed)
    best, best_cost = None, float("inf")
    for _ in range(n_init):
        centres = x[torch.randint(len(x), (1,), generator=gen)]
        for _ in range(k - 1):  # k-means++: next centre far from the current ones
            d2 = torch.cdist(x, centres).min(dim=1).values ** 2
            pick = torch.multinomial((d2 / d2.sum()).cpu(), 1, generator=gen)
            centres = torch.cat([centres, x[pick]])
        for _ in range(n_iter):
            label = torch.cdist(x, centres).argmin(dim=1)  # (N,)
            one_hot = torch.nn.functional.one_hot(label, k).to(x.dtype)  # (N, k)
            counts = one_hot.sum(0).clamp_min(1)
            new = torch.einsum("ik,ia->ka", one_hot, x) / counts[:, None]  # cluster means
            if torch.allclose(new, centres):
                break
            centres = new
        cost = (torch.cdist(x, centres).min(dim=1).values ** 2).sum().item()
        if cost < best_cost:
            best, best_cost = label, cost
    return best


def spectral_embedding(w: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Ng–Jordan–Weiss: the k lowest eigenvectors of L_sym with every row scaled to unit length.
    Returns the eigenvalues (N,) and the embedding (N, k)."""
    evals, evecs = spectrum(normalized_laplacian(w))
    u = evecs[:, :k]
    return evals, u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)


def spectral_clustering(w: torch.Tensor, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eigenvectors → k-means in the embedding. Returns labels (N,), eigenvalues of L_sym (N,) and embedding (N, k)."""
    evals, u = spectral_embedding(w, k)
    return kmeans(u, k, seed), evals, u


def fiedler_vector(w: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Second-smallest eigenpair of the random-walk Laplacian D⁻¹L, from L_sym: f = D^{-1/2} u₂."""
    evals, evecs = spectrum(normalized_laplacian(w))
    f = evecs[:, 1] * degrees(w).clamp_min(1e-12).rsqrt()
    return evals[1].item(), f / f.abs().max()


def adjusted_rand_index(a: torch.Tensor, b: torch.Tensor) -> float:
    """ARI (Hubert & Arabie): 1 for identical partitions up to renaming, ≈ 0 for random ones."""
    table = torch.einsum("ia,ib->ab", torch.nn.functional.one_hot(a).double(), torch.nn.functional.one_hot(b).double())

    def pairs(m: torch.Tensor) -> torch.Tensor:
        return (m * (m - 1) / 2).sum()

    n_pairs = pairs(torch.tensor(float(len(a)), dtype=torch.float64))
    index, rows, cols = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = rows * cols / n_pairs
    return ((index - expected) / (0.5 * (rows + cols) - expected)).item()
