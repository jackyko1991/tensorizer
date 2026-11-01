"""k-means with several restarts at once (batched tensor ops), and a small t-SNE map trained with autograd."""

import torch

from problems._common.progress import steps

from .equation import assign, inertia, sq_distances, tsne_affinities, tsne_loss, update_centroids


def kmeans(y: torch.Tensor, k: int, n_init: int, n_iter: int, seed: int) -> dict:
    """Lloyd's algorithm for R = n_init random starts in parallel; keep the start with the lowest final inertia.

    Each start picks k random cells as its first centres. Every iteration: assign each cell to its nearest centre,
    then move each centre to the mean of its cells. Neither step can increase the inertia.
    Returns the best run's centres (T+1, K, P) and labels (T+1, C) per iteration, and all inertias (T+1, R).
    """
    gen = torch.Generator().manual_seed(seed)
    C = y.shape[0]
    starts = torch.stack([torch.randperm(C, generator=gen)[:k] for _ in range(n_init)]).to(y.device)  # (R, K)
    mu = y[starts]  # (R, K, P)
    centres, labels, inertias = [], [], []
    for it in steps(n_iter + 1, "k-means"):
        d = sq_distances(y, mu)  # (R, C, K)
        A = assign(d)  # (R, C, K)
        centres.append(mu)
        labels.append(d.argmin(-1))
        inertias.append(inertia(d, A))
        if it < n_iter:
            mu = update_centroids(y, A, mu)
    inertias = torch.stack(inertias)  # (T+1, R)
    best = int(inertias[-1].argmin())
    return {
        "centres": torch.stack(centres)[:, best],  # (T+1, K, P)
        "labels": torch.stack(labels)[:, best],  # (T+1, C)
        "inertia": inertias,  # (T+1, R)
        "best": best,
    }


def tsne(y: torch.Tensor, perplexity: float, n_iter: int, seed: int) -> torch.Tensor:
    """2D t-SNE map of the PC scores y (C, P), optimised with Adam; starts from PC1–PC2 (scaled small)."""
    P = tsne_affinities(y, perplexity)
    Y0 = y[:, :2] / y[:, :2].std() * 1e-2
    gen = torch.Generator().manual_seed(seed)
    Y = (Y0 + 1e-4 * torch.randn(Y0.shape, generator=gen).to(y.device)).requires_grad_(True)
    opt = torch.optim.Adam([Y], lr=1.0)
    for it in steps(n_iter, "t-SNE"):
        loss = tsne_loss(Y, P, exaggeration=12.0 if it < n_iter // 4 else 1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return Y.detach()
