"""The UMAP pipeline on tensors: kNN graph → fuzzy simplicial set → spectral start → layout optimisation, with the
forces written by hand (batched over the whole edge list) or taken from autograd on the sampled loss; plus PCA and
Procrustes alignment for the comparisons."""

import time

import torch

from problems._common.progress import steps

from . import equation as E


def fuzzy_graph(x: torch.Tensor, k: int, block: int | None = 1024) -> dict:
    """The high-dimensional graph: kNN (N, k), ρ and σ (N,), directed memberships P and their union W (N, N)."""
    dist, idx = E.knn(x, k, block=block)
    rho, sigma = E.smooth_knn(dist, k)
    p = E.membership(dist, rho, sigma)  # (N, k)
    P = torch.zeros(len(x), len(x), dtype=x.dtype, device=x.device).scatter_(1, idx, p)  # p_{j|i} placed at (i, j)
    return {"dist": dist, "idx": idx, "rho": rho, "sigma": sigma, "P": P, "W": E.fuzzy_union(P)}


def spectral_init(w: torch.Tensor, dim: int = 2, seed: int = 0) -> torch.Tensor:
    """The dim smallest non-trivial eigenvectors of the normalised graph Laplacian L = I - D^{-1/2} W D^{-1/2}
    (the trivial one, D^{1/2} 1, is shifted out of the way by adding 2 v0 v0ᵀ), mapped back by D^{-1/2} and
    rescaled to [0, 10] per axis as umap-learn does. torch.linalg.eigh is dense, O(N³): seconds for a few thousand
    points (umap-learn uses a sparse eigensolver). A disconnected graph has several zero eigenvalues; their
    eigenvectors are then component indicators, which simply place the components apart. Fallback if eigh fails: a
    uniform random start in [0, 10]², as umap-learn's init="random"."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(w)
    try:
        dm = w.sum(1).clamp(min=1e-12).rsqrt()  # (N,) D^{-1/2}
        lap = torch.eye(n, dtype=w.dtype, device=w.device) - dm[:, None] * w * dm[None, :]
        v0 = 1 / dm / (1 / dm).norm()  # (N,) the trivial eigenvector D^{1/2} 1 (eigenvalue 0)
        _, vec = torch.linalg.eigh(lap + 2 * v0[:, None] * v0[None, :])  # trivial one moved to the top, (N, N)
        y = dm[:, None] * vec[:, :dim]  # Laplacian eigenmaps: L_rw y = λ y with y = D^{-1/2} v
    except torch.linalg.LinAlgError:
        print("spectral initialisation failed; random start instead")
        y = torch.rand(n, dim, generator=g).to(w)
    y = y + 1e-4 * (y.abs().max()) * torch.randn(n, dim, generator=g).to(w)  # break exact ties
    return 10 * (y - y.min(0).values) / (y.max(0).values - y.min(0).values)


def edge_list(w: torch.Tensor, n_epochs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Heads, tails (E,) of the non-zero memberships (both directions) and epochs per sample max(w)/w_ij: an edge of
    weight w_ij is sampled w_ij/max(w) times per epoch on average; edges too weak to be sampled once are dropped."""
    i, j = torch.nonzero(w, as_tuple=True)
    wij = w[i, j]
    keep = wij >= wij.max() / n_epochs
    return i[keep], j[keep], wij.max() / wij[keep]


def optimise(y0, i, j, eps, a, b, n_epochs, neg_rate=5, seed=0, mode="forces", w=None, snaps=0):
    """SGD on the layout. Each epoch every edge whose turn has come is active (a (E,) mask: the same schedule as
    umap-learn), each active edge draws neg_rate random negatives, and ALL points move at once (a Jacobi update of
    the batch, where umap-learn updates edge by edge). Learning rate α = 1 - epoch/n_epochs.
    mode "forces": the hand-written force formula; mode "autograd": the gradient of the same sampled loss from
    autograd, clipped per coordinate to ±4 after summing. Returns Y (N, 2), snapshots [(epoch, Y, C)], seconds."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(y0)
    y = y0.clone()
    nxt = eps.clone()  # epoch of the next sample of each edge
    every = max(1, n_epochs // snaps) if snaps else 0
    frames = [(0, y.cpu().clone())] if snaps else []
    t0 = time.perf_counter()
    for ep in steps(n_epochs, f"layout ({mode})"):
        alpha = 1.0 - ep / n_epochs
        act = nxt <= ep  # (E,) edges sampled this epoch
        ia, ja = i[act], j[act]
        i_neg = ia.repeat_interleave(neg_rate)  # (S,) heads of the negative samples
        k_neg = torch.randint(0, n, (len(i_neg),), generator=g).to(y.device)  # (S,) random tails
        if mode == "forces":
            y += alpha * E.forces(y, ia, ja, i_neg, k_neg, a, b)
        else:
            yy = y.clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(E.sampled_loss(yy, ia, ja, i_neg, k_neg, a, b, eps=1e-3), yy)
            y -= alpha * grad.clamp(-4, 4)
        nxt[act] += eps[act]
        if every and ((ep + 1) % every == 0 or ep + 1 == n_epochs):
            frames.append((ep + 1, y.cpu().clone()))
    secs = time.perf_counter() - t0
    frames = [(e, f, float(E.cross_entropy(w, f.to(w.device), a, b))) for e, f in frames]  # untimed: for the plot
    return y, frames, secs


def umap(x, k=15, min_dist=0.1, n_epochs=500, seed=0, mode="forces", snaps=0, neg_rate=5, prep=None) -> dict:
    """The whole pipeline on (N, D) data; returns the layout and every intermediate, with wall time per stage.
    prep: an earlier result on the same data whose graph and spectral start are reused (with their times)."""
    torch.manual_seed(seed)
    if prep is None:
        t = {}
        t0 = time.perf_counter()
        graph = fuzzy_graph(x, k)
        t["graph"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        y0 = spectral_init(graph["W"], 2, seed)
        t["spectral"] = time.perf_counter() - t0
    else:
        graph = {key: prep[key] for key in ("dist", "idx", "rho", "sigma", "P", "W")}
        y0, t = prep["Y0"], {key: prep["seconds"][key] for key in ("graph", "spectral")}
    a, b = E.fit_ab(min_dist)
    i, j, eps = edge_list(graph["W"], n_epochs)
    y, frames, t["layout"] = optimise(y0, i, j, eps, a, b, n_epochs, neg_rate, seed, mode, graph["W"], snaps)
    t["total"] = t["graph"] + t["spectral"] + t["layout"]
    return {"Y": y, "Y0": y0, "frames": frames, "a": a, "b": b, "edges": len(i), "seconds": t, **graph}


def expected_step(y, w, a, b, lr):
    """One step along the EXPECTED force: every ordered pair attracts with weight w_ij and repels with weight
    2(1 - w_ij) (the repulsion of a pair moves only its head, the reversed pair moves the other end). Without
    clipping this is exactly -lr ∂C/∂y of the full cross-entropy."""
    n = len(y)
    i, j = torch.nonzero(~torch.eye(n, dtype=torch.bool, device=y.device), as_tuple=True)
    wij = w[i, j]
    return y + lr * E.forces(y, i, j, i, j, a, b, clip=None, eps=0.0, w_att=wij, w_rep=2 * (1 - wij))


def pca(x: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """Projection on the first principal axes: X_c = X - mean = U S Vᵀ (SVD), coordinates X_c V[:, :dim]."""
    xc = x - x.mean(0)
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    return xc @ vh[:dim].T


def procrustes(y: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Align y to ref by translation, uniform scale, rotation and reflection (orthogonal R from the SVD of
    y_cᵀ ref_c); both normalised to unit Frobenius norm. Returns y aligned in ref's frame and the disparity
    ‖y R - ref‖² ∈ [0, 1] (0 = same shape)."""
    yc, rc = y - y.mean(0), ref - ref.mean(0)
    ny, nr = yc.norm(), rc.norm()
    u, s, vh = torch.linalg.svd((yc / ny).T @ (rc / nr))
    r = u @ vh  # (2, 2) orthogonal
    disparity = float(1 - s.sum() ** 2)
    return (yc / ny) @ r * s.sum() * nr + ref.mean(0), disparity
