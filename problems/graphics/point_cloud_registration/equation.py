r"""Geometry → tensor translation: the PCL registration pipeline as batched tensor operations.

A rigid motion maps a point x to R x + t (R a rotation, t a translation). Indices: b = batch (hypotheses or ICP
starts), n = point, k = neighbour, i, j = 3D axes, h = histogram bin.

    rigid motion     y_{bni} = R_{bij} x_{bnj} + t_{bi}                         j summed
    Kabsch           H_{ij} = w_n (p_{ni} - p̄_i)(q_{nj} - q̄_j), H = U S Vᵀ, R = V diag(1, 1, det V Uᵀ) Uᵀ, t = q̄ - R p̄
    k-NN             d_{nm} = |x_n - y_m|  (torch.cdist), then the k smallest per row (topk)
    voxel grid       key_n = ⌊x_n / s⌋,  c_v = mean of the x_n with key_n = v               (scatter-mean)
    outlier removal  m_n = mean_k d_{nk},  keep m_n ≤ mean(m) + α std(m)
    normals          C_{nij} = (1/K) Σ_k (x_{nki} - x̄_{ni})(x_{nkj} - x̄_{nj}),  C v = λ v,  normal = v of the smallest λ
    curvature        σ_n = λ_0 / (λ_0 + λ_1 + λ_2)
    Darboux frame    u = n_p,  v = u × d̂,  w = u × v,  d̂ = (q - p)/|q - p|
    pair features    α = v·n_q,  φ = u·d̂,  θ = atan2(w·n_q, u·n_q)                → binned into 3 × 11 histograms
    point-to-plane   r_n = (R x_n + t - q_n)·n_n ≈ r_n + (x_n × n_n)·ω + n_n·δt   (small rotation ω)
"""

import torch

BINS = 11  # bins per FPFH angle histogram (PCL's FPFHSignature33 = 3 × 11)


def rodrigues(omega: torch.Tensor) -> torch.Tensor:
    """Axis-angle ω (..., 3) → rotation R (..., 3, 3) = I + sin θ K + (1 - cos θ) K², θ = |ω|, K = [ω/θ]×."""
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-15)  # (..., 1)
    k = omega / theta
    zero = torch.zeros_like(k[..., 0])
    K = torch.stack(
        [
            torch.stack([zero, -k[..., 2], k[..., 1]], -1),
            torch.stack([k[..., 2], zero, -k[..., 0]], -1),
            torch.stack([-k[..., 1], k[..., 0], zero], -1),
        ],
        -2,
    )  # (..., 3, 3): K v = k × v
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    return eye + s * K + (1 - c) * (K @ K)


def transform(R: torch.Tensor, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """y_{bni} = R_{bij} x_{bnj} + t_{bi}   (j summed).   R (..., 3, 3), t (..., 3), X (..., N, 3) → (..., N, 3)."""
    return torch.einsum("...ij,...nj->...ni", R, X) + t[..., None, :]


def compose(R1, t1, R2, t2):
    """(R1, t1) after (R2, t2): x ↦ R1 (R2 x + t2) + t1 = (R1 R2) x + (R1 t2 + t1)."""
    return R1 @ R2, torch.einsum("...ij,...j->...i", R1, t2) + t1


def invert(R, t):
    """The inverse motion: x = Rᵀ (y - t), i.e. (Rᵀ, -Rᵀ t)."""
    Rt = R.transpose(-1, -2)
    return Rt, -torch.einsum("...ij,...j->...i", Rt, t)


def rotation_error_deg(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angle of the rotation R1ᵀ R2 in degrees: cos θ = (tr(R1ᵀ R2) - 1) / 2.   (...,)."""
    cos = ((torch.einsum("...ij,...ij->...", R1, R2) - 1) / 2).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def kabsch(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor | None = None):
    """Best rigid motion (least squares) taking the points P onto Q, batched over the leading dimensions.

    P, Q (..., N, 3), weights w (..., N) (default 1). Returns R (..., 3, 3), t (..., 3) minimising Σ w |R p + t - q|².
    """
    if w is None:
        w = torch.ones(P.shape[:-1], dtype=P.dtype, device=P.device)
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-12)  # (..., N), sums to 1
    p_bar = torch.einsum("...n,...ni->...i", w, P)  # weighted centroids (..., 3)
    q_bar = torch.einsum("...n,...ni->...i", w, Q)
    H = torch.einsum("...n,...ni,...nj->...ij", w, P - p_bar[..., None, :], Q - q_bar[..., None, :])  # n summed
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    d = torch.sign(torch.linalg.det(V @ U.transpose(-1, -2)))  # -1 would be a reflection: flip the last axis
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = V @ D @ U.transpose(-1, -2)
    return R, q_bar - torch.einsum("...ij,...j->...i", R, p_bar)


def knn(X: torch.Tensor, Y: torch.Tensor, k: int, chunk: int = 2048):
    """The k nearest points of Y for every point of X: distances and indices (N, k), nearest first.

    torch.cdist builds a chunk of the distance matrix d_{nm} = |x_n - y_m| at a time, so memory stays O(chunk × M).
    """
    out_d, out_i = [], []
    for s in range(0, len(X), chunk):  # a loop over chunks of rows, not over points
        d = torch.cdist(X[s : s + chunk], Y, compute_mode="use_mm_for_euclid_dist")  # (chunk, M)
        if k == 1:
            d, i = d.min(dim=1, keepdim=True)  # same as topk(1), faster
        else:
            d, i = d.topk(k, dim=1, largest=False)
        out_d.append(d)
        out_i.append(i)
    return torch.cat(out_d), torch.cat(out_i)


def voxel_downsample(X: torch.Tensor, size: float):
    """VoxelGrid filter: key_n = ⌊x_n / s⌋, one output point per occupied voxel = the mean of its points.

    Returns the centroids (V, 3) and the number of points in each voxel (V,).
    """
    keys = torch.floor(X / size).long()  # (N, 3) integer voxel coordinates
    _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(counts), 3, dtype=X.dtype, device=X.device).index_add_(0, inverse, X)  # scatter-add
    return sums / counts[:, None], counts


def statistical_outlier_mask(X: torch.Tensor, k: int = 20, alpha: float = 1.0) -> torch.Tensor:
    """StatisticalOutlierRemoval: keep the points whose mean distance to their k neighbours is at most
    mean + α·std of that quantity over the cloud.   → bool (N,)."""
    d, _ = knn(X, X, k + 1)  # the first neighbour is the point itself (distance 0)
    m = d[:, 1:].mean(1)  # (N,) mean neighbour distance
    return m <= m.mean() + alpha * m.std()


def estimate_normals(X: torch.Tensor, k: int, viewpoint: torch.Tensor):
    """NormalEstimation: the covariance of each point's k neighbours, its smallest eigenvector, oriented toward
    the viewpoint; and the surface variation λ_0 / (λ_0 + λ_1 + λ_2).   → normals (N, 3), curvature (N,)."""
    _, idx = knn(X, X, k)  # (N, k), includes the point itself
    nb = X[idx]  # (N, k, 3)
    centred = nb - nb.mean(1, keepdim=True)
    C = torch.einsum("nki,nkj->nij", centred, centred) / k  # (N, 3, 3), k summed
    lam, vec = torch.linalg.eigh(C)  # eigenvalues ascending
    normals = vec[:, :, 0]  # (N, 3) direction of least spread
    flip = ((viewpoint - X) * normals).sum(1) < 0  # PCL's flipNormalTowardsViewpoint
    normals = torch.where(flip[:, None], -normals, normals)
    curvature = lam[:, 0].clamp(min=0) / lam.sum(1).clamp(min=1e-30)
    return normals, curvature


def pair_features(p, n_p, q, n_q):
    """Darboux-frame angles (α, φ, θ) for point pairs (p, q) with normals; all inputs (..., 3) → three (...)."""
    d = q - p
    dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    d_hat = d / dist
    u = n_p
    v = torch.linalg.cross(u, d_hat)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    w = torch.linalg.cross(u, v)
    alpha = (v * n_q).sum(-1)  # ∈ [-1, 1]
    phi = (u * d_hat).sum(-1)  # ∈ [-1, 1]
    theta = torch.atan2((w * n_q).sum(-1), (u * n_q).sum(-1))  # ∈ [-π, π]
    return alpha, phi, theta, dist[..., 0]


def _histogram(values: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """(N, k) values → (N, BINS) counts normalised to sum 100 per row (PCL's convention)."""
    b = ((values - lo) / (hi - lo) * BINS).floor().long().clamp(0, BINS - 1)
    h = torch.nn.functional.one_hot(b, BINS).to(values.dtype).sum(1)  # (N, BINS)
    return 100 * h / h.sum(1, keepdim=True)


def fpfh(X: torch.Tensor, N: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Fast Point Feature Histograms (simplified): (N, 33) = three 11-bin histograms of α, φ, θ per point.

    SPFH_p = histograms over the pairs (p, q_k) of p's k neighbours; FPFH_p = SPFH_p + (1/k) Σ_k SPFH_{q_k} / ω_k with
    ω_k = |q_k - p|, each 11-bin block renormalised to 100.
    """
    _, idx = knn(X, X, k + 1)
    idx = idx[:, 1:]  # (N, k) neighbours, self removed
    alpha, phi, theta, dist = pair_features(X[:, None], N[:, None], X[idx], N[idx])  # each (N, k)
    spfh = torch.cat([_histogram(alpha, -1, 1), _histogram(phi, -1, 1), _histogram(theta, -torch.pi, torch.pi)], 1)
    wgt = 1 / dist.clamp(min=1e-12)  # (N, k)
    f = spfh + torch.einsum("nk,nkh->nh", wgt, spfh[idx]) / k  # (N, 33), k summed
    blocks = f.reshape(-1, 3, BINS)
    return (100 * blocks / blocks.sum(-1, keepdim=True)).reshape(-1, 3 * BINS)


def point_to_plane_system(x: torch.Tensor, q: torch.Tensor, n: torch.Tensor, w: torch.Tensor):
    """Linearised point-to-plane least squares for the small motion ξ = (ω, δt), batched over b:
    A_{bn·} = [x × n, n], r_{bn} = (x - q)·n, then H = Aᵀ W A (6 × 6) and g = Aᵀ W r.   → H (B, 6, 6), g (B, 6)."""
    A = torch.cat([torch.linalg.cross(x, n), n], -1)  # (B, N, 6)
    r = ((x - q) * n).sum(-1)  # (B, N)
    return torch.einsum("bn,bni,bnj->bij", w, A, A), torch.einsum("bn,bni,bn->bi", w, A, r)
